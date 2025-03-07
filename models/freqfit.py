'''
Adaptive Fourier Neural Operators: Efficient Token Mixers for Transformers
https://arxiv.org/pdf/2111.13587.pdf
'''
import math
import torch
import torch.fft
import torch.nn as nn


def init_ssf_scale_shift(blocks, dim):
    scale = nn.Parameter(torch.ones(blocks, dim))
    shift = nn.Parameter(torch.zeros(blocks, dim))

    nn.init.normal_(scale, mean=1, std=.02)
    nn.init.normal_(shift, std=.02)

    return scale, shift


def ssf_ada(x, scale, shift):
    assert scale.shape == shift.shape
    if x.shape[-1] == scale.shape[0]:
        return x * scale + shift
    elif x.shape[1] == scale.shape[0]:
        return x * scale.view(1, -1, 1, 1) + shift.view(1, -1, 1, 1)
    else:
        raise ValueError('the input tensor shape does not match the shape of the scale factor.')


class GlobalFilter(nn.Module):
    '''
    https://github.com/NVlabs/AFNO-transformer/blob/master/afno/gfn.py
    '''
    def __init__(self, blocks, dim, h=14):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(blocks, h, dim, 2, dtype=torch.float32) * 0.02)
        self.h = h
        self.ssf_scale, self.ssf_shift = init_ssf_scale_shift(blocks, dim)

    def forward(self, block, x, spatial_size=None, dim=1):
        B, a, C = x.shape


        x = x.to(torch.float32)
        res = x
        x = torch.fft.rfft(x, dim=dim, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight[block].squeeze())
        x = x * weight
        x = torch.fft.irfft(x, n=a, dim=dim, norm='ortho')
        x = ssf_ada(x, self.ssf_scale[block], self.ssf_shift[block])
        x = x + res
        return x



class GlobalFilter2D(nn.Module):
    '''
    https://github.com/NVlabs/AFNO-transformer/blob/master/afno/gfn.py
    '''
    def __init__(self, blocks, dim, h=14, w=8):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(blocks, h, w, dim, 2, dtype=torch.float32) * 0.02)
        self.h = h
        self.w = w
        self.ssf_scale, self.ssf_shift = init_ssf_scale_shift(blocks, dim)

    def forward(self, block, x, spatial_size=None):
        B, N, C = x.shape
        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size

        x = x.view(B, a, b, C)

        x = x.to(torch.float32)
        res = x

        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight[block].squeeze())
        x = x * weight
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')
        x = ssf_ada(x, self.ssf_scale[block], self.ssf_shift[block])
        x = x + res
        x = x.reshape(B, N, C)
        return x

class GlobalFilter2D_real(nn.Module):
    '''
    https://github.com/NVlabs/AFNO-transformer/blob/master/afno/gfn.py
    '''
    def __init__(self, blocks, dim, h=14, w=8):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(blocks, h, w, dim, dtype=torch.float32) * 0.02)
        self.h = h
        self.w = w
        self.ssf_scale, self.ssf_shift = init_ssf_scale_shift(blocks, dim)

    def forward(self, block, x, spatial_size=None):
        B, N, C = x.shape
        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size

        x = x.view(B, a, b, C)
        x = x.to(torch.float32)
        res = x

        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        x = x * self.complex_weight[block].squeeze()
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')
        x = ssf_ada(x, self.ssf_scale[block], self.ssf_shift[block])
        x = x + res
        x = x.reshape(B, N, C)
        return x
