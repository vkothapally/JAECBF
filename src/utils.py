import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import os
import signal
import functional as FC
from tensor import ComplexTensor


def init_kernel(frame_len,frame_hop,num_fft=None,window="sqrt_hann"):
    if window != "sqrt_hann":
        raise RuntimeError("Now only support sqrt hanning window in order "
                           "to make signal perfectly reconstructed")
    fft_size = 2 ** math.ceil(math.log2(frame_len)) if not num_fft else num_fft
    window = torch.hann_window(frame_len) ** 0.5
    S_ = 0.5 * (fft_size * fft_size / frame_hop) ** 0.5
    w  = torch.fft.rfft(torch.eye(fft_size) / S_)
    kernel = torch.stack([w.real, w.imag],-1)
    kernel = torch.transpose(kernel, 0, 2) * window
    kernel = torch.reshape(kernel, (fft_size + 2, 1, frame_len))
    return kernel


class STFTBase(nn.Module):
    def __init__(self,frame_len,frame_hop,window="sqrt_hann",num_fft=None):
        super(STFTBase, self).__init__()
        K = init_kernel(frame_len,frame_hop,num_fft=num_fft,window=window)
        self.K = nn.Parameter(K, requires_grad=False)
        self.stride = frame_hop
        self.window = window

    def freeze(self): self.K.requires_grad = False
    def unfreeze(self): self.K.requires_grad = True
    def check_nan(self):
        num_nan = torch.sum(torch.isnan(self.K))
        if num_nan:
            raise RuntimeError(
                "detect nan in STFT kernels: {:d}".format(num_nan))
    def extra_repr(self):
        return "window={0}, stride={1}, requires_grad={2}, kernel_size={3[0]}x{3[2]}".format(self.window, self.stride, self.K.requires_grad, self.K.shape)


class STFT(STFTBase):
    def __init__(self, *args, **kwargs):
        super(STFT, self).__init__(*args, **kwargs)

    def forward(self, x):
        if x.dim() not in [2, 3]:
            print(x.shape)
            raise RuntimeError("Expect 2D/3D tensor, but got {:d}D".format(
                x.dim()))
        self.check_nan()
        if x.dim() == 2:
            x = torch.unsqueeze(x, 1)
        c = F.conv1d(x, self.K, stride=self.stride, padding=0)
        r, i = torch.chunk(c, 2, dim=1)
        m = (r ** 2 + i ** 2) ** 0.5
        p = torch.atan2(i, r)
        return m, p, r, i


class iSTFT(STFTBase):
    def __init__(self, *args, **kwargs):
        super(iSTFT, self).__init__(*args, **kwargs)

    def forward(self, m, p, squeeze=False):
        if p.dim() != m.dim() or p.dim() not in [2, 3]:
            raise RuntimeError("Expect 2D/3D tensor, but got {:d}D".format(
                p.dim()))
        self.check_nan()
        if p.dim() == 2:
            p = torch.unsqueeze(p, 0)
            m = torch.unsqueeze(m, 0)
        r = m * torch.cos(p)
        i = m * torch.sin(p)
        c = torch.cat([r, i], dim=1)
        s = F.conv_transpose1d(c, self.K, stride=self.stride, padding=0)
        if squeeze:
            s = torch.squeeze(s)
        return s

class Conv1D(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super(Conv1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else th.unsqueeze(x, 1))
        if squeeze:
            x = th.squeeze(x)
        return x

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        self._name = 'Identity'
    def forward(self, x): return x

class SelectItem(nn.Module):
    def __init__(self, item_index):
        super(SelectItem, self).__init__()
        self._name = 'selectitem'
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]

class ChannelWiseLayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super(ChannelWiseLayerNorm, self).__init__(*args, **kwargs)

    def forward(self, x):
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        x = torch.transpose(x, 1, 2)
        x = super(ChannelWiseLayerNorm, self).forward(x)
        x = torch.transpose(x, 1, 2)
        return x

class MultiChannelSTFT(nn.Module):
    def __init__(self, num_fft=512, frame_len=512, frame_hop=256):
        super(MultiChannelSTFT, self).__init__()
        self.init_mic_pos()
        self.stft = STFT(frame_len=frame_len, frame_hop=frame_hop, num_fft=num_fft)
        self.num_bins = num_fft//2+1

    def init_mic_pos(self):
        #radius = 0.035
        mic1 = [0.13, 0] #2-2-2-14-2-2-2
        mic2 = [0.11, 0]
        mic3 = [0.09, 0]
        mic4 = [0.07, 0]
        mic5 = [-0.07, 0]
        mic6 = [-0.09, 0]
        mic7 = [-0.11, 0]
        mic8 = [-0.13, 0]
        mic_position = np.array([mic1, mic2, mic3, mic4, mic5, mic6, mic7, mic8])
        self.mic_position = mic_position
        self.n_mic = self.mic_position.shape[0]
        

    def forward(self, all):
        """
        [0] x    - input mixture waveform, with shape [batch_size (B), n_channel (M), seq_len (S)]
        [1] echo - input mixture waveform, with shape [batch_size (B), seq_len (S)]
        """
        x    = all[0]
        echo = all[1]
        
        batch_size, n_channel, S_ = x.shape
        all_s = x.contiguous().view(-1, S_)
        magnitude, phase, real, imag = self.stft(all_s)
        _, F_, K_ = phase.shape
        phase = phase.view(batch_size, n_channel, F_, K_)
        magnitude = magnitude.view(batch_size, n_channel, F_, K_)
        real = real.view(batch_size,n_channel,F_,K_)
        imag = imag.view(batch_size,n_channel,F_,K_)

        magnitude_echo, phase_echo, real_echo, imag_echo = self.stft(echo)
        real_echo = real_echo.unsqueeze(1)
        imag_echo = imag_echo.unsqueeze(1)

        return real, imag, real_echo, imag_echo


def apply_cRM_filter(cRM_filter: ComplexTensor,
                     mix: ComplexTensor) -> ComplexTensor:
    es = FC.einsum('bftd,bcfdt->bcft', [cRM_filter.conj(), mix])
    return es


def  get_power_spectral_density_matrix_self_with_cm_t(xs: ComplexTensor, mask: ComplexTensor = None,
                                      averaging=True,
                                      normalization=True,
                                      eps: float = 1e-6
                                      ) -> ComplexTensor:
    psd = FC.einsum('...ct,...et->...tce', [xs, xs.conj()])
    return psd 


def apply_beamforming_vector(beamform_vector: ComplexTensor,
                             mix: ComplexTensor) -> ComplexTensor:
    es = FC.einsum('bftc,bfct->bft', [beamform_vector.conj(), mix])
    return es

if __name__ == '__main__':
    x      = torch.randn(10,8,64000)
    echo   = torch.randn(10,1,64000)
    feat_extraction = MultiChannelSTFT()
    real, imag, real_echo, imag_echo = feat_extraction([x, echo])

    istft = iSTFT(frame_len=512,frame_hop=256,num_fft=512)
    y = istft(real[:,0], imag[:,0])


    print(x.shape, echo.shape)
    print(real.shape)
    print(real_echo.shape)
    print(y.shape)






