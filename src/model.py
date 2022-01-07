# This code is a part of Tencent AI's Intellectual property
#----------------------------------------------------------------------------------------------------------
# Title: JOINT AEC AND BEAMFORMING with Double-Talk Detection using RNN-Transformer
# Authors: Vinay Kothapally, Yong Xu, Meng Yu, Shi-Xiong Zhang, Dong Yu
# Submitted to ICASPP 2022
#----------------------------------------------------------------------------------------------------------
# This Script is currently private and only meant to serve as a reference to the reviewers of ICASPP 2022
# This work has been conducted by Vinay Kothapally during his Internship at Tencent AI Lab
#----------------------------------------------------------------------------------------------------------



import numpy as np
import torch as th
import torch.nn as nn
import functional as FC
from utils import *
from utils import iSTFT
from tensor import ComplexTensor
import torch.nn.functional as Fn


th.manual_seed(1187)
np.random.seed(1187)

def param(nnet, Mb=True):
    """
    Return number parameters(not bytes) in nnet
    """
    neles = sum([param.nelement() for param in nnet.parameters()])
    return neles / 10**6 if Mb else neles


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, output_dim, locations=[None,None], dropout=0.0, bias=True, batch_first=True):
        super(SelfAttention, self).__init__()
        self.name    = 'SelfAttention'
        self.locations = locations
        self.attn    = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.lnorm1  = nn.LayerNorm(embed_dim)
        self.linear  = nn.Linear(embed_dim, output_dim)
        self.lnorm2  = nn.LayerNorm(output_dim)
        
    def compute_mask(self, dim, locations, device):
        mask = torch.ones(dim, dim).to(device)
        locations  = [dim*(-1)**(i+1) if locations[i]==None else locations[i] for i in range(len(locations))]
        mask = mask - torch.tril(mask, locations[0]-1) - torch.triu(mask, locations[1]+1)
        mask = mask.masked_fill(mask==0, -1e10)
        mask = mask.masked_fill(mask==1, 0.0)
        return mask

    def forward(self, q, k, v):
        # query, key, value ----> [batch, frequency, time]
        b,f,t     = q.shape
        q,k,v = q.permute(2,0,1), k.permute(2,0,1), v.permute(2,0,1) # [batch, frequency, time] ----> [time, batch, frequency]
        
        attn_mask = self.compute_mask(t, self.locations, q.device)
        attn_output, attn_output_weights = self.attn(q, k, v, attn_mask=attn_mask)
        attn_output = self.lnorm1((v + attn_output).permute(1,0,2))
        attn_output = Fn.relu(self.lnorm2(self.linear(attn_output))).transpose(1,2)
        return attn_output


class DTD_Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, output_dim, locations=[None,None], dropout=0.0, bias=True, batch_first=True):
        super(DTD_Attention, self).__init__()
        self.name    = 'DTD_Attention'
        self.locations = locations
        self.attn    = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.gru     = nn.Sequential(*[nn.GRU(embed_dim,embed_dim,1,batch_first=True),SelectItem(0)])
        self.linear  = nn.Linear(embed_dim, 1)
        
        
    def compute_mask(self, dim, locations, device):
        mask = torch.ones(dim, dim).to(device)
        locations  = [dim*(-1)**(i+1) if locations[i]==None else locations[i] for i in range(len(locations))]
        mask = mask - torch.tril(mask, locations[0]-1) - torch.triu(mask, locations[1]+1)
        mask = mask.masked_fill(mask==0, -1e10)
        mask = mask.masked_fill(mask==1, 0.0)
        return mask

    def forward(self, q, k, v):
        # query, key, value ----> [batch, frequency, time, nchannels]
        b,f,t,e   = q.shape
        q, k, v   = q.permute(2,0,3,1).reshape(t,b*e,f), k.permute(2,0,3,1).reshape(t,b*e,f), v.permute(2,0,3,1).reshape(t,b*e,f) # [time, batch, embedding]
        attn_mask = self.compute_mask(t, self.locations, q.device)
        attn_output = self.attn(q, k, v, attn_mask=attn_mask)[0]
        attn_output = attn_output.reshape(t,b,e,f).permute(1,3,0,2)                   # Attn Out ----> [batch, frequency, time, nchannels]
        gruOut      = self.gru(v.permute(1,0,2)).reshape(t,b,f,e).permute(1,0,3,2)    # GRU Out -----> [batch, time, nchannels, frequency]
        double_talk = th.sigmoid(self.linear(gruOut)).permute(0,3,1,2)                # DTD maksing -> [batch, 1, time, nchannels] mask for each channel
        dtd_attn    = double_talk * attn_output
        return dtd_attn


class RNNBF(nn.Module):
    def __init__(
            self, L=512, N=256, X=8, R=4, B=256, H=512, P=3, F=256, cos=True, ipd=""):
        super(RNNBF, self).__init__()

        self.cos = cos
        self.modelname = 'Joint-AEC-and-Beamforming-DTD-RNN-Transformer'
        self.df_computer = MultiChannelSTFT()
        dim_conv = 257
        self.nmics = X
        self.ntaps = 2
        
        
        # Feature Extraction -----------------------------------------------------------------------------------------------------
        self.spatial_idx     = th.tril_indices(self.nmics+1,self.nmics+1)
        self.attn_spatial    = SelfAttention(embed_dim=(self.nmics+1)*(self.nmics+2), num_heads=2, output_dim=(self.nmics+2), locations=[None,None], dropout=0.2)
        self.conv1x1_1       = Conv1D((self.nmics+2)*(B+1), B, 1)
        # ------------------------------------------------------------------------------------------------------------------------
        
        
        # Multi-Channel AEC Using (Mixture + EchoRef) ----------------------------------------------------------------------------
        self.gru_encoder     = nn.Sequential(*[nn.GRU(B,B,2,batch_first=True),SelectItem(0)]) 
        self.cRM_mix_real    = Conv1D(B, (self.nmics*self.df_computer.num_bins)*self.ntaps, 1)
        self.cRM_mix_imag    = Conv1D(B, (self.nmics*self.df_computer.num_bins)*self.ntaps, 1)
        self.cRM_echo_real   = Conv1D(B, self.df_computer.num_bins*self.ntaps, 1)
        self.cRM_echo_imag   = Conv1D(B, self.df_computer.num_bins*self.ntaps, 1)
        # ------------------------------------------------------------------------------------------------------------------------
        

        # Speech and Noise PSD Computation ---------------------------------------------------------------------------------------
        self.aecfeat_idx     = th.tril_indices(2*self.nmics+1,2*self.nmics+1)
        self.attn_aecout     = SelfAttention(embed_dim=306, num_heads=3, output_dim=(self.nmics+2), locations=[None,None], dropout=0.2)
        self.conv1x1_2       = Conv1D((self.nmics+2)*(B+1), B, 1)
        self.gru_target      = nn.Sequential(*[nn.GRU(B,B,1,batch_first=True),SelectItem(0)]) 
        self.gru_noise       = nn.Sequential(*[nn.GRU(B,B,1,batch_first=True),SelectItem(0)]) 
        self.conv1x1_speech_mask_mix_real   = Conv1D(B, self.df_computer.num_bins*self.ntaps, 3, 1, 1) # speech and noise cRM filter weights
        self.conv1x1_speech_mask_mix_imag   = Conv1D(B, self.df_computer.num_bins*self.ntaps, 3, 1, 1)
        self.conv1x1_noise_mask_mix_real    = Conv1D(B, self.df_computer.num_bins*self.ntaps, 3, 1, 1)
        self.conv1x1_noise_mask_mix_imag    = Conv1D(B, self.df_computer.num_bins*self.ntaps, 3, 1, 1)
        # ------------------------------------------------------------------------------------------------------------------------
        
        
        
        # MVDR Beamforming -------------------------------------------------------------------------------------------------------
        n_mic=8
        self.hid_dim=256
        self.both_ln     = nn.LayerNorm([(2*n_mic+1) * (2*n_mic+1) * 2 *2],elementwise_affine=True)
        self.Dense_pca1  = nn.Linear( (2*n_mic+1) * (2*n_mic+1) * 2 * 2, self.hid_dim)
        self.GRU_pca_h1  = nn.GRU(self.hid_dim, self.hid_dim, 2, batch_first=True)
        self.attn_Gpca1  = SelfAttention(embed_dim=self.hid_dim, num_heads=4, output_dim=self.hid_dim, locations=[None,None], dropout=0.2)
        self.Dense_pca2  = nn.Linear(self.hid_dim, (2*n_mic+1) * 2)        
        self.dtd_attn    = DTD_Attention(embed_dim=self.df_computer.num_bins, num_heads=1, output_dim=self.df_computer.num_bins, locations=[None,None], dropout=0.2)
        self.istft       = iSTFT(frame_len=L, frame_hop=L // 2, num_fft=L)
        # ------------------------------------------------------------------------------------------------------------------------
        


        print('-'*90)
        print('Model                   : '+self.modelname)
        print('Trainable params.       : ',str(np.round(param(self, Mb=True),2))+' M')
        print('-'*90)

    def get_lps(self, real,imag):
        magnitude = real**2 + imag**2
        lps = th.log(magnitude + 1e-8) #4-th channel from right to left
        mean_lps=th.mean(lps,-1, keepdim=True)
        lps=lps-mean_lps
        return lps

    def permute_mask(self, mask):
        b,f_taps,t = mask.shape 
        mask = th.transpose(mask,1,2)  # B x 514 x T ---> B x T x 514
        mask = mask.reshape(b,t,self.df_computer.num_bins,self.ntaps) # B x T X 257 x 2
        mask = th.transpose(mask,1,2)  # B x 257 x T x 2
        return mask

    def compute_ipd(self, phase):
        '''phase [B, M, F, K], return IPD [B, I, F, K]'''
        mic_pairs = [[0, 7], [1, 6], [0, 5], [2, 7], [2, 3], [3, 4]]
        self.ipd_left = [t[0] for t in mic_pairs]
        self.ipd_right = [t[1] for t in mic_pairs]
        cos_ipd = torch.cos(phase[:, self.ipd_left] - phase[:, self.ipd_right])
        cos_ipd = cos_ipd.reshape(cos_ipd.shape[0], -1, cos_ipd.shape[3])
        return cos_ipd

    def apply_df_filters(self, filter, real, imag, t_roll=[-1,0,1], f_roll=[-1,0,1], channels=False):
        real_tf_shift = th.stack([th.roll(real,(i,j),dims=(2,3)) for i in t_roll for j in f_roll],4).transpose(-1,-2)
        imag_tf_shift = th.stack([th.roll(imag,(i,j),dims=(2,3)) for i in t_roll for j in f_roll],4).transpose(-1,-2)
        imag_tf_shift += 1e-10
        
        y_complex = ComplexTensor(real_tf_shift, imag_tf_shift) #[B,C,F,T]
        if channels == True:
            est_complex = FC.einsum('bcftd,befdt->bcft', [filter.conj(), y_complex])
        else:
            est_complex = apply_cRM_filter(filter, y_complex) #[B,C,F,T]
        est_real_part = est_complex.real 
        est_imag_part = est_complex.imag + 1.0e-10
        return est_complex, est_real_part, est_imag_part

    def covariance_matrix(self, xs, ntaps, mode='temporal'):
        b,c,f,t = xs.shape
        dimension = 3 if mode == 'temporal' else 2
        def roll(x, ntaps):
            real_shift = th.stack([th.roll(x.real,i,dims=dimension) for i in range(ntaps)],4).transpose(-1,-2)
            imag_shift = th.stack([th.roll(x.imag,i,dims=dimension) for i in range(ntaps)],4).transpose(-1,-2)
            return ComplexTensor(real_shift, imag_shift)
        xs  = roll(xs, ntaps)
        rxx = FC.einsum('...ct,...et->...tce', [xs, xs.conj()])[...,self.temporal_idx[0],self.temporal_idx[1]]  #.flatten(dim=4)
        rxx = rxx.permute(0,2,3,1,4).reshape(b,f,t,-1) # B,C,F,T,rxx --> B,F,T,C*rxx
        return rxx
    
    def forward(self, x, echo, aecout=False, verbose=False):
        if x.dim() not in [2, 3]:
            raise RuntimeError(
                "{} accept 2/3D tensor as input, but got {:d}".format(
                    self.__name__, x.dim()))
        
        # when inference, only one utt
        if x.dim() == 2:
            x = th.unsqueeze(x, 0)
            spk_num = th.unsqueeze(spk_num, 0)
            directions = th.unsqueeze(directions, 0)
       
        real, imag, real_echo, imag_echo = self.df_computer([x, echo])
        
        #----------------------------------------------------------------------------------------------------------------------------------------------------
        # Feature Extraction - Covariance Matirx of (Mixture + EchoRef)
        #----------------------------------------------------------------------------------------------------------------------------------------------------
        if verbose: print('*'*90)
        if verbose: print('Input Audio Shape            : ', x.shape)
        if verbose: print('*'*90)
        
        if verbose: print('Mixture, Echo Ref            : ', real.shape, real_echo.shape)
        real_mix_echo, imag_mix_echo = th.cat((real,real_echo),1), th.cat((imag,imag_echo),1)
        cplx_mix_echo = ComplexTensor(real_mix_echo, imag_mix_echo)
        if verbose: print('Complex Mic+Echo Sig.        : ', cplx_mix_echo.real.shape, cplx_mix_echo.imag.shape)
        

        spatial_psd    = get_power_spectral_density_matrix_self_with_cm_t(cplx_mix_echo.permute(0,2,1,3))[...,self.spatial_idx[0],self.spatial_idx[1]] 
        spatial_psd    = th.cat((spatial_psd.real,spatial_psd.imag),-1)
        b,f,t,e        = spatial_psd.shape
        spatial_psd    = spatial_psd.reshape(b*f,t,e).transpose(1,2)
        spatial_psd  = self.conv1x1_1(self.attn_spatial(spatial_psd,spatial_psd,spatial_psd).reshape(b,-1,t))
        if verbose: print('Mix+Echo Spatial PSD         : ', spatial_psd.shape)

        #----------------------------------------------------------------------------------------------------------------------------------------------------
        # Deep Learning based Multi-Channel Acoustic Echo Canceller --- Outputs [mix h1*mix h2*echo] (8+8+1--> 17 channels)
        #----------------------------------------------------------------------------------------------------------------------------------------------------
        if verbose: print('-'*60)
        if verbose: print('Multi-Channel Acoustic Echo Cancellation')
        if verbose: print('-'*60)
        spatial_feats  = self.gru_encoder(spatial_psd.transpose(1,2)).transpose(1,2)
        if verbose: print('Spatial Encoded Feats        : ', spatial_feats.shape)
        
        # Apply 8-channel filters for the echo_ref too and sum 8 channels of mixture and echo_ref -- Dr. Xu
        b,f,t = spatial_feats.shape
        f = f+1
        cRM_mix_real = self.cRM_mix_real(spatial_feats).transpose(1,2).reshape(b,t,f,self.nmics,-1).permute(0,3,2,1,4)
        cRM_mix_imag = self.cRM_mix_imag(spatial_feats).transpose(1,2).reshape(b,t,f,self.nmics,-1).permute(0,3,2,1,4)
        cRM_mix      = ComplexTensor(cRM_mix_real,cRM_mix_imag)
        if verbose: print('[AEC] cRF for  Mix           : ', cRM_mix.real.shape,cRM_mix.imag.shape)
        
        cRM_echo_real = self.cRM_echo_real(spatial_feats).transpose(1,2).reshape(b,t,f,1,-1).permute(0,3,2,1,4)
        cRM_echo_imag = self.cRM_echo_imag(spatial_feats).transpose(1,2).reshape(b,t,f,1,-1).permute(0,3,2,1,4)
        cRM_echo      = ComplexTensor(cRM_echo_real, cRM_echo_imag)
        if verbose: print('[AEC] cRF for  Echo          : ', cRM_echo.real.shape, cRM_echo.imag.shape)

        cplx_aec_mix,  aec_mix_real, aec_mix_imag   =  self.apply_df_filters(cRM_mix,  real, imag, t_roll=[-1,0], f_roll=[0], channels=True)
        cplx_aec_echo, aec_echo_real, aec_echo_imag =  self.apply_df_filters(cRM_echo, real_echo, imag_echo, t_roll=[-1,0], f_roll=[0], channels=True)
        if verbose: print('[AEC Out] Mix_h, Echo_h      : ', aec_mix_real.shape, aec_echo_real.shape)
        
        mix_aecout_real = th.cat((real, aec_mix_real, aec_echo_real),1) #.permute(0,2,3,1)
        mix_aecout_imag = th.cat((imag, aec_mix_imag, aec_echo_imag),1) #.permute(0,2,3,1)
        
        if verbose: print('[Mix, Mix_h, Echo_h]         : ', mix_aecout_real.shape, mix_aecout_imag.shape)
        real_aec, imag_aec = mix_aecout_real, (mix_aecout_imag + 1e-10)
        cplx_aec = ComplexTensor(real_aec, imag_aec)

        #----------------------------------------------------------------------------------------------------------------------------------------------------
        # Deep Filtering based cRM filter estimation --- Used to compute Speech and Noise Covariane Matrices (17x17)
        #----------------------------------------------------------------------------------------------------------------------------------------------------
        if verbose: print('-'*60)
        if verbose: print('Joint Spatial RNN-Beamformer')
        if verbose: print('-'*60)

        audiofeats   = get_power_spectral_density_matrix_self_with_cm_t(cplx_aec.permute(0,2,1,3))[...,self.aecfeat_idx[0],self.aecfeat_idx[1]] 
        audiofeats   = th.cat((audiofeats.real,audiofeats.imag),-1)
        b,f,t,e      = audiofeats.shape
        audiofeats   = audiofeats.reshape(b*f,t,e).transpose(1,2)
        audiofeats   = self.attn_aecout(audiofeats,audiofeats,audiofeats)
        audiofeats   = self.conv1x1_2(audiofeats.reshape(b,-1,t))
        if verbose: print('JRNN-BF Input                : ', mix_aecout_real.shape, mix_aecout_imag.shape)
        if verbose: print('Spatial Features + Attention : ', audiofeats.shape)

        if verbose: print(' ')
        if verbose: print('Multi-Channel Speech/Noise PSD Computation')
        if verbose: print('-'*45)
        # Inputs to CRM Filter Design (y_targ, y_noise) : B x 128 x T
        y_targ       = self.gru_target(audiofeats.transpose(1,2)).transpose(1,2)    
        y_noise      = self.gru_noise(audiofeats.transpose(1,2)).transpose(1,2)    

        # Deep Filter-01 for Mixture signals -------------------- SPEECH ---------------------
        cRM_speech_mask = ComplexTensor(self.permute_mask(self.conv1x1_speech_mask_mix_real(y_targ)), self.permute_mask(self.conv1x1_speech_mask_mix_imag(y_targ))+ 1e-10)
        # Deep Filter-01 for Mixture signals -------------------- NOISE ---------------------
        cRM_noise_mask = ComplexTensor(self.permute_mask(self.conv1x1_noise_mask_mix_real(y_noise)), self.permute_mask(self.conv1x1_noise_mask_mix_imag(y_noise))+ 1e-10)
        if verbose: print('cRM - Speech/Noise Est       : ', cRM_speech_mask.real.shape, cRM_speech_mask.imag.shape)
        
        
        _, est_speech_real,     est_speech_imag        = self.apply_df_filters(cRM_speech_mask, real_aec, imag_aec, t_roll=[-1,0], f_roll=[0])
        _, est_noise_real,      est_noise_imag         = self.apply_df_filters(cRM_noise_mask,  real_aec, imag_aec, t_roll=[-1,0], f_roll=[0])
        
        est_speech_cplx = ComplexTensor(est_speech_real, est_speech_imag).transpose(1,2)
        est_noise_cplx  = ComplexTensor(est_noise_real,  est_noise_imag).transpose(1,2)

        if verbose: print('Multi-Ch Speech/Noise Est.   : ', (est_speech_cplx.real.shape, est_speech_cplx.imag.shape))
        
        speech_PSD = get_power_spectral_density_matrix_self_with_cm_t(est_speech_cplx)
        noise_PSD  = get_power_spectral_density_matrix_self_with_cm_t(est_noise_cplx) #[B,F,T,C,C]
        if verbose: print('Speech/Noise PSD Matrix      : ', (speech_PSD.real.shape, speech_PSD.imag.shape))
        #----------------------------------------------------------------------------------------------------------------------------------------------------
        



        #----------------------------------------------------------------------------------------------------------------------------------------------------
        # Deep Learning based Beamforming weight Computation
        #----------------------------------------------------------------------------------------------------------------------------------------------------
        if verbose: print(' ')
        if verbose: print('Beamformer Weight Computation')
        if verbose: print('-'*30)
        
        speech_PSD      = th.cat((speech_PSD.real.flatten(-2), speech_PSD.imag.flatten(-2)),dim=-1)
        noise_PSD       = th.cat((noise_PSD.real.flatten(-2),  noise_PSD.imag.flatten(-2)), dim=-1)
        PSDs_flatten    = self.both_ln(th.cat([noise_PSD,speech_PSD],dim=-1))
        if verbose: print('PSDs Combined + Flat + LNorm : ', PSDs_flatten.shape)

        ws_per_frame    = Fn.leaky_relu(self.Dense_pca1(PSDs_flatten))
        if verbose: print('Linear PCA [Dim. Reduction]  : ', ws_per_frame.shape)

        b,f,t,e = ws_per_frame.shape
        ws_per_frame    = self.GRU_pca_h1(ws_per_frame.reshape(b*f,t,self.hid_dim))[0].reshape(b,f,t,self.hid_dim)
        if verbose: print('GRU on Low Dim. Features     : ', ws_per_frame.shape)

        # Self-Attention for GRU Encoded features 
        b,f,t,e = ws_per_frame.shape
        ws_per_frame    = ws_per_frame.reshape(b*f,t,e).transpose(1,2)
        ws_per_frame    = self.attn_Gpca1(ws_per_frame,ws_per_frame,ws_per_frame)
        ws_per_frame    = ws_per_frame.transpose(1,2).reshape(b,f,t,e)
        if verbose: print('SA on PCA-GRU Features       : ', ws_per_frame.shape)

        ws_per_frame    = self.Dense_pca2(ws_per_frame)
        # Self-Attention for GRU Encoded features 
        b,f,t,e = ws_per_frame.shape
        # ws_per_frame    = ws_per_frame.reshape(b*f,t,e).transpose(1,2)
        ws_per_frame    = self.dtd_attn(ws_per_frame,ws_per_frame,ws_per_frame)
        # ws_per_frame    = ws_per_frame.transpose(1,2).reshape(b,f,t,e)
        if verbose: print('RNN-based DTST Module        : ', ws_per_frame.shape)

        ws_per_frame    = ComplexTensor(ws_per_frame[:,:,:,:est_noise_cplx.size(2)],ws_per_frame[:,:,:,est_noise_cplx.size(2):])
        if verbose: print('JRNN_AEC_BF_DTDT Weights     : ', ws_per_frame.real.shape, ws_per_frame.imag.shape)

        cplx_bf_input = cplx_aec.transpose(1,2)
        if verbose: print('Cplx Mic. + AEC processed    : ', cplx_bf_input.real.shape, cplx_bf_input.imag.shape)

        bf_enhanced = apply_beamforming_vector(ws_per_frame, cplx_bf_input) # mc_complex (B,F,C*2,T)
        if verbose: print('JRNN_AEC_BF_DTDT Output      : ', (bf_enhanced.real.shape, bf_enhanced.imag.shape))

        
        #----------------------------------------------------------------------------------------------------------------------------------------------------
        # Single-Channel Audio Synthesis 
        #----------------------------------------------------------------------------------------------------------------------------------------------------
        bf_enhanced = ComplexTensor(bf_enhanced.real, bf_enhanced.imag+1.0e-10)
        bf_enhanced_mag, bf_enhanced_phase = bf_enhanced.abs(), bf_enhanced.angle()
        

        est = self.istft(bf_enhanced_mag, bf_enhanced_phase, squeeze=False)
        if verbose: print('*'*90)
        if verbose: print('Output Audio Shape      : ', est.shape)
        if verbose: print('*'*90)
        
        return est, bf_enhanced_mag
        
      

if __name__=='__main__':
    model = RNNBF().to('cuda')
    x           = th.Tensor(np.random.randn(1,8,64000)).to('cuda')
    echo        = th.Tensor(np.random.randn(1,64000,)).to('cuda')
    est, bf_enhanced_mag = model(x, echo, verbose=True)
    print('--------------------------------- Script Inputs and Outputs :: Summary')
    print('Input Mix audio  : ', x.shape)
    print('Input echo ref   : ', echo.shape)
    print('Output Estimated : ', est.shape)
    print('Output Est Mag   : ', bf_enhanced_mag.shape)
    print('--------------------------------------------------------------------------\n')
    print('Done!')

    
