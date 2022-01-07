import numbers
from typing import Union, List

import numpy
import torch


class ComplexTensor:
    def __init__(self, real: Union[torch.Tensor, numpy.ndarray], imag=None):
        if imag is None:
            if isinstance(real, numpy.ndarray):
                if real.dtype.kind == 'c':
                    imag = real.imag
                    real = real.real
                else:
                    imag = numpy.zeros_like(real)
            else:
                imag = torch.zeros_like(real)

        if isinstance(real, numpy.ndarray):
            real = torch.from_numpy(real)
        if isinstance(imag, numpy.ndarray):
            imag = torch.from_numpy(imag)

        if not torch.is_tensor(real):
            raise TypeError(
                f'The first arg must be torch.Tensor'
                f'but got {type(real)}')

        if not torch.is_tensor(imag):
            raise TypeError(
                f'The second arg must be torch.Tensor'
                f'but got {type(imag)}')
        if not real.size() == imag.size():
            raise ValueError(f'The two inputs must have same sizes: '
                             f'{real.size()} != {imag.size()}')

        self.real = real
        self.imag = imag

    def __getitem__(self, item) -> 'ComplexTensor':
        return ComplexTensor(self.real[item], self.imag[item])

    def __setitem__(self, item, value: Union['ComplexTensor',
                                             torch.Tensor, numbers.Number]):
        if isinstance(value, (ComplexTensor, complex)):
            self.real[item] = value.real
            self.imag[item] = value.imag
        else:
            self.real[item] = value
            self.imag[item] = 0

    def __mul__(self,
                other: Union['ComplexTensor', torch.Tensor, numbers.Number]) \
            -> 'ComplexTensor':
        if isinstance(other, (ComplexTensor, complex)):
            return ComplexTensor(
                self.real * other.real - self.imag * other.imag,
                self.real * other.imag + self.imag * other.real)
        else:
            return ComplexTensor(self.real * other, self.imag * other)

    def __rmul__(self,
                 other: Union['ComplexTensor', torch.Tensor, numbers.Number]) \
            -> 'ComplexTensor':
        if isinstance(other, (ComplexTensor, complex)):
            return ComplexTensor(
                other.real * self.real - other.imag * self.imag,
                other.imag * self.real + other.real * self.imag)
        else:
            return ComplexTensor(other * self.real, other * self.imag)

    def __imul__(self, other):
        if isinstance(other, (ComplexTensor, numbers.Complex)):
            t = self * other
            self.real = t.real
            self.imag = t.imag
        else:
            self.real *= other
            self.imag *= other
        return self

    def __truediv__(self, other) -> 'ComplexTensor':
        if isinstance(other, (ComplexTensor, complex)):
            den = other.real ** 2 + other.imag ** 2
            return ComplexTensor(
                (self.real * other.real + self.imag * other.imag) / den,
                (-self.real * other.imag + self.imag * other.real) / den)
        else:
            return ComplexTensor(self.real / other, self.imag / other)

    def __rtruediv__(self, other) -> 'ComplexTensor':
        if isinstance(other, (ComplexTensor, complex)):
            den = self.real ** 2 + self.imag ** 2
            return ComplexTensor(
                (other.real * self.real + other.imag * self.imag) / den,
                (-other.real * self.imag + other.imag * self.real) / den)
        else:
            den = self.real ** 2 + self.imag ** 2
            return ComplexTensor(other * self.real / den,
                                 -other * self.imag / den)

    def __itruediv__(self, other) -> 'ComplexTensor':
        if isinstance(other, (ComplexTensor, numbers.Complex)):
            t = self / other
            self.real = t.real
            self.imag = t.imag
        else:
            self.real /= other
            self.imag /= other
        return self

    def __add__(self, other) -> 'ComplexTensor':
        if isinstance(other, (ComplexTensor, complex)):
            return ComplexTensor(self.real + other.real,
                                 self.imag + other.imag)
        else:
            return ComplexTensor(self.real + other, self.imag)

    def __radd__(self, other) -> 'ComplexTensor':
        if isinstance(other, (ComplexTensor, complex)):
            return ComplexTensor(other.real + self.real,
                                 other.imag + self.imag)
        else:
            return ComplexTensor(other + self.real, self.imag)

    def __iadd__(self, other) -> 'ComplexTensor':
        if isinstance(other, (ComplexTensor, complex)):
            self.real += other.real
            self.imag += other.imag
        else:
            self.real += other
        return self

    def __sub__(self, other) -> 'ComplexTensor':
        if isinstance(other, (ComplexTensor, complex)):
            return ComplexTensor(self.real - other.real,
                                 self.imag - other.imag)
        else:
            return ComplexTensor(self.real - other, self.imag)

    def __rsub__(self, other) -> 'ComplexTensor':
        if isinstance(other, (ComplexTensor, complex)):
            return ComplexTensor(other.real - self.real,
                                 other.imag - self.imag)
        else:
            return ComplexTensor(other - self.real, self.imag)

    def __isub__(self, other) -> 'ComplexTensor':
        if isinstance(other, (ComplexTensor, complex)):
            self.real -= other.real
            self.imag -= other.imag
        else:
            self.real -= other
        return self

    def __matmul__(self, other) -> 'ComplexTensor':
        if isinstance(other, ComplexTensor):
            o_real = torch.matmul(self.real, other.real) - \
                torch.matmul(self.imag, other.imag)
            o_imag = torch.matmul(self.real, other.imag) + \
                torch.matmul(self.imag, other.real)
        else:
            o_real = torch.matmul(self.real, other)
            o_imag = torch.matmul(self.imag, other)
        return ComplexTensor(o_real, o_imag)

    def __rmatmul__(self, other) -> 'ComplexTensor':
        if isinstance(other, ComplexTensor):
            o_real = torch.matmul(other.real, self.real) - \
                torch.matmul(other.imag, self.imag)
            o_imag = torch.matmul(other.real, self.imag) + \
                torch.matmul(other.imag, self.real)
        else:
            o_real = torch.matmul(other, self.real)
            o_imag = torch.matmul(other, self.imag)
        return ComplexTensor(o_real, o_imag)

    def __imatmul__(self, other) -> 'ComplexTensor':
        if isinstance(other, (ComplexTensor, numbers.Complex)):
            t = self @ other
            self.real = t.real
            self.imag = t.imag
        else:
            self.real @= other
            self.imag @= other
        return self

    def __neg__(self) -> 'ComplexTensor':
        return ComplexTensor(-self.real, -self.imag)

    def __eq__(self, other) -> torch.Tensor:
        if isinstance(other, (ComplexTensor, complex)):
            return (self.real == other.real) ** (self.imag == other.imag)
        else:
            return (self.real == other) ** (self.imag == 0)

    def __len__(self) -> int:
        return len(self.real)

    def __repr__(self) -> str:
        return 'ComplexTensor(\nReal:\n' \
               + repr(self.real) + '\nImag:\n' + repr(self.imag) + '\n)'

    def __abs__(self) -> torch.Tensor:
        return (self.real * self.real + self.imag * self.imag).sqrt()

    def __pow__(self, exponent) -> 'ComplexTensor':
        if exponent == -2:
            return 1 / (self * self)
        if exponent == -1:
            return 1 / self
        if exponent == 0:
            return ComplexTensor(torch.ones_like(self.real))
        if exponent == 1:
            return self.clone()
        if exponent == 2:
            return self * self

        _abs = self.abs().pow(exponent)
        _angle = exponent * self.angle()
        return ComplexTensor(_abs * torch.cos(_angle),
                             _abs * torch.sin(_angle))

    def __ipow__(self, exponent) -> 'ComplexTensor':
        c = self ** exponent
        self.real = c.real
        self.imag = c.imag
        return self

    def abs(self) -> torch.Tensor:
        return (self.real * self.real + self.imag * self.imag).sqrt()

    def angle(self) -> torch.Tensor:
        return torch.atan2(self.imag, self.real)

    def backward(self) -> None:
        self.real.backward()
        self.imag.backward()

    def byte(self) -> 'ComplexTensor':
        return ComplexTensor(self.real.byte(), self.imag.byte())

    def clone(self) -> 'ComplexTensor':
        return ComplexTensor(self.real.clone(), self.imag.clone())

    def flatten(self, dim) -> 'ComplexTensor':
        return ComplexTensor(torch.flatten(self.real,dim), torch.flatten(self.imag,dim))

    def reshape(self, *shape) -> 'ComplexTensor':
        return ComplexTensor(self.real.reshape(shape), self.imag.reshape(shape))
    
    def zeromean(self, dim) -> 'ComplexTensor':
        self.real = self.real - torch.mean(self.real, dim=dim, keepdim=True)
        self.imag = self.imag - torch.mean(self.imag, dim=dim, keepdim=True)
        return ComplexTensor(self.real, self.imag)

    def conj(self) -> 'ComplexTensor':
        return ComplexTensor(self.real, -self.imag)

    def conj_(self) -> 'ComplexTensor':
        self.imag.neg_()
        return self

    def contiguous(self) -> 'ComplexTensor':
        return ComplexTensor(self.real.contiguous(), self.imag.contiguous())

    def copy_(self) -> 'ComplexTensor':
        self.real = self.real.copy_()
        self.imag = self.imag.copy_()
        return self

    def cpu(self) -> 'ComplexTensor':
        return ComplexTensor(self.real.cpu(), self.imag.cpu())

    def cuda(self) -> 'ComplexTensor':
        return ComplexTensor(self.real.cuda(), self.imag.cuda())

    def expand(self, *sizes):
        return ComplexTensor(self.real.expand(*sizes),
                             self.imag.expand(*sizes))

    def expand_as(self, *args, **kwargs):
        return ComplexTensor(self.real.expand_as(*args, **kwargs),
                             self.imag.expand_as(*args, **kwargs))

    def detach(self) -> 'ComplexTensor':
        return ComplexTensor(self.real.detach(), self.imag.detach())

    def detach_(self) -> 'ComplexTensor':
        self.real.detach_()
        self.imag.detach_()
        return self

    @property
    def device(self):
        assert self.real.device == self.imag.device
        return self.real.device

    def diag(self) -> 'ComplexTensor':
        return ComplexTensor(self.real.diag(), self.imag.diag())

    def diagonal(self) -> 'ComplexTensor':
        return ComplexTensor(self.real.diag(), self.imag.diag())

    def dim(self) -> int:
        return self.real.dim()

    def double(self) -> 'ComplexTensor':
        return ComplexTensor(self.real.double(), self.imag.double())

    @property
    def dtype(self) -> torch.dtype:
        return self.real.dtype

    def eq(self, other) -> torch.Tensor:
        if isinstance(other, (ComplexTensor, complex)):
            return (self.real == other.real) * (self.imag == other.imag)
        else:
            return (self.real == other) * (self.imag == 0)

    def equal(self, other) -> bool:
        if isinstance(other, (ComplexTensor, complex)):
            return self.real.equal(other.real) and self.imag.equal(other.imag)
        else:
            return self.real.equal(other) and self.imag.equal(0)

    def float(self) -> 'ComplexTensor':
        return ComplexTensor(self.real.float(), self.imag.float())

    def fill(self, value) -> 'ComplexTensor':
        if isinstance(value, complex):
            return ComplexTensor(self.real.fill(value.real),
                                 self.imag.fill(value.imag))
        else:
            return ComplexTensor(self.real.fill(value), self.imag.fill(0))

    def fill_(self, value) -> 'ComplexTensor':
        if isinstance(value, complex):
            self.real.fill_(value.real)
            self.imag.fill_(value.imag)
        else:
            self.real.fill_(value)
            self.imag.fill_(0)
        return self

    def gather(self, dim, index) -> 'ComplexTensor':
        return ComplexTensor(self.real.gather(dim, index),
                             self.real.gather(dim, index))

    def get_device(self, *args, **kwargs):
        return self.real.get_device(*args, **kwargs)

    def half(self) -> 'ComplexTensor':
        return ComplexTensor(self.real.half(), self.imag.half())

    def index_add(self, dim, index, tensor) -> 'ComplexTensor':
        return ComplexTensor(self.real.index_add(dim, index, tensor),
                             self.imag.index_add(dim, index, tensor))

    def index_copy(self, dim, index, tensor) -> 'ComplexTensor':
        return ComplexTensor(self.real.index_copy(dim, index, tensor),
                             self.imag.index_copy(dim, index, tensor))

    def index_fill(self, dim, index, value) -> 'ComplexTensor':
        return ComplexTensor(self.real.index_fill(dim, index, value),
                             self.imag.index_fill(dim, index, value))

    def index_select(self, dim, index) -> 'ComplexTensor':
        return ComplexTensor(self.real.index_select(dim, index),
                             self.imag.index_select(dim, index))

    def inverse(self, ntry=5):
        # m x n x n
        in_size = self.size()
        a = self.view(-1, self.size(-1), self.size(-1))
        # see "The Matrix Cookbook" (http://www2.imm.dtu.dk/pubdb/p.php?3274)
        # "Section 4.3"
        for i in range(ntry):
            t = i * 0.1

            e = a.real + t * a.imag
            f = a.imag - t * a.real

            try:
                x = torch.matmul(f, e.inverse())
                z = (e + torch.matmul(x, f)).inverse()
            except Exception:
                if i == ntry - 1:
                    raise
                continue

            if t != 0.:
                eye = torch.eye(a.real.size(-1), dtype=a.real.dtype,
                                device=a.real.device)[None]
                o_real = torch.matmul(z, (eye - t * x))
                o_imag = -torch.matmul(z, (t * eye + x))
            else:
                o_real = z
                o_imag = -torch.matmul(z, x)

            o = ComplexTensor(o_real, o_imag)
            return o.view(*in_size)

    def item(self) -> numbers.Number:
        return self.real.item() + 1j * self.imag.item()

    def masked_fill(self, mask, value) -> 'ComplexTensor':
        if isinstance(value, complex):
            return ComplexTensor(self.real.masked_fill(mask, value.real),
                                 self.imag.masked_fill(mask, value.imag))

        else:
            return ComplexTensor(self.real.masked_fill(mask, value),
                                 self.imag.masked_fill(mask, 0))

    def masked_fill_(self, mask, value) -> 'ComplexTensor':
        if isinstance(value, complex):
            self.real.masked_fill_(mask, value.real)
            self.imag.masked_fill_(mask, value.imag)
        else:
            self.real.masked_fill_(mask, value)
            self.imag.masked_fill_(mask, 0)
        return self

    def mean(self, *args, **kwargs) -> 'ComplexTensor':
        return ComplexTensor(self.real.mean(*args, **kwargs),
                             self.imag.mean(*args, **kwargs))

    def neg(self) -> 'ComplexTensor':
        return ComplexTensor(-self.real, -self.imag)

    def neg_(self) -> 'ComplexTensor':
        self.real.neg_()
        self.imag.neg_()
        return self

    def nelement(self) -> int:
        return self.real.nelement()

    def numel(self) -> int:
        return self.real.numel()

    def new(self, *args, **kwargs) -> 'ComplexTensor':
        return ComplexTensor(self.real.new(*args, **kwargs),
                             self.imag.new(*args, **kwargs))

    def new_empty(self, size, dtype=None, device=None, requires_grad=False) \
            -> 'ComplexTensor':
        real = self.real.new_empty(size,
                                   dtype=dtype,
                                   device=device,
                                   requires_grad=requires_grad)
        imag = self.imag.new_empty(size,
                                   dtype=dtype,
                                   device=device,
                                   requires_grad=requires_grad)
        return ComplexTensor(real, imag)

    def new_full(self, size, fill_value, dtype=None, device=None,
                 requires_grad=False) -> 'ComplexTensor':
        if isinstance(fill_value, complex):
            real_value = fill_value.real
            imag_value = fill_value.imag
        else:
            real_value = fill_value
            imag_value = 0.

        real = self.real.new_full(size,
                                  fill_value=real_value,
                                  dtype=dtype,
                                  device=device,
                                  requires_grad=requires_grad)
        imag = self.imag.new_full(size,
                                  fill_value=imag_value,
                                  dtype=dtype,
                                  device=device,
                                  requires_grad=requires_grad)
        return ComplexTensor(real, imag)

    def new_tensor(self, data, dtype=None, device=None,
                   requires_grad=False) -> 'ComplexTensor':
        if isinstance(data, ComplexTensor):
            real = data.real
            imag = data.imag
        elif isinstance(data, numpy.ndarray):
            if data.dtype.kind == 'c':
                real = data.real
                imag = data.imag
            else:
                real = data
                imag = None
        else:
            real = data
            imag = None

        real = self.real.new_tensor(real,
                                    dtype=dtype,
                                    device=device,
                                    requires_grad=requires_grad)
        if imag is None:
            imag = torch.zeros_like(real,
                                    dtype=dtype,
                                    device=device,
                                    requires_grad=requires_grad)
        else:
            imag = self.imag.new_tensor(imag,
                                        dtype=dtype,
                                        device=device,
                                        requires_grad=requires_grad)
        return ComplexTensor(real, imag)

    def numpy(self) -> numpy.ndarray:
        return self.real.numpy() + 1j * self.imag.numpy()

    def permute(self, *dims) -> 'ComplexTensor':
        return ComplexTensor(self.real.permute(*dims),
                             self.imag.permute(*dims))

    def pow(self, exponent) -> 'ComplexTensor':
        return self ** exponent

    def requires_grad_(self) -> 'ComplexTensor':
        self.real.requires_grad_()
        self.imag.requires_grad_()
        return self

    @property
    def requires_grad(self):
        assert self.real.requires_grad == self.imag.requires_grad
        return self.real.requires_grad

    @requires_grad.setter
    def requires_grad(self, value):
        self.real.requires_grad = value
        self.imag.requires_grad = value

    def repeat(self, *sizes):
        return ComplexTensor(self.real.repeat(*sizes),
                             self.imag.repeat(*sizes))

    def retain_grad(self) -> 'ComplexTensor':
        self.real.retain_grad()
        self.imag.retain_grad()
        return self

    def share_memory_(self) -> 'ComplexTensor':
        self.real.share_memory_()
        self.imag.share_memory_()
        return self

    @property
    def shape(self) -> torch.Size:
        return self.real.shape

    def size(self, *args, **kwargs) -> torch.Size:
        return self.real.size(*args, **kwargs)

    def sqrt(self) -> 'ComplexTensor':
        return self ** 0.5

    def squeeze(self, dim) -> 'ComplexTensor':
        return ComplexTensor(self.real.squeeze(dim),
                             self.imag.squeeze(dim))

    def sum(self, *args, **kwargs) -> 'ComplexTensor':
        return ComplexTensor(self.real.sum(*args, **kwargs),
                             self.imag.sum(*args, **kwargs),)

    def take(self, indices) -> 'ComplexTensor':
        return ComplexTensor(self.real.take(indices), self.imag.take(indices))

    def to(self, *args, **kwargs) -> 'ComplexTensor':
        return ComplexTensor(self.real.to(*args, **kwargs),
                             self.imag.to(*args, **kwargs))

    def tolist(self) -> List[numbers.Number]:
        return [r + 1j * i
                for r, i in zip(self.real.tolist(), self.imag.tolist())]

    def transpose(self, dim0, dim1) -> 'ComplexTensor':
        return ComplexTensor(self.real.transpose(dim0, dim1),
                             self.imag.transpose(dim0, dim1))

    def transpose_(self, dim0, dim1) -> 'ComplexTensor':
        self.real.transpose_(dim0, dim1)
        self.imag.transpose_(dim0, dim1)
        return self

    def type(self) -> str:
        return self.real.type()

    def unfold(self, dim, size, step):
        return ComplexTensor(self.real.unfold(dim, size, step),
                             self.imag.unfold(dim, size, step))

    def unsqueeze(self, dim) -> 'ComplexTensor':
        return ComplexTensor(self.real.unsqueeze(dim),
                             self.imag.unsqueeze(dim))

    def unsqueeze_(self, dim) -> 'ComplexTensor':
        self.real.unsqueeze_(dim)
        self.imag.unsqueeze_(dim)
        return self

    def view(self, *args, **kwargs) -> 'ComplexTensor':
        return ComplexTensor(self.real.view(*args, **kwargs),
                             self.imag.view(*args, **kwargs))

    def view_as(self, tensor):
        return self.view(tensor.size())

    
