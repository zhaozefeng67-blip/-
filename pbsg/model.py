# -*- coding: utf-8 -*-
"""One fixed Graph-WaveNet-style backbone (section 6.1).  Only A is ever replaced.

Everything else is frozen across methods: input projection, dilated temporal convolutions,
channel width, spatial propagation operator (``A @ H``), residual/skip wiring, prediction
head, loss and optimisation.  No original global adaptive adjacency survives in the
proposed branch -- the only graph that reaches the propagation is the builder output.
"""
import torch
import torch.nn as nn

from .builders import spmm


class GWNBackbone(nn.Module):
    """Dilated temporal stack with per-layer spatial propagation.

    ``dilations`` is explicit so the receptive field is auditable: with kernel size 2 and
    no padding, ``RF = 1 + sum(dilations)`` and ``T_after = tin - sum(dilations)``.  The
    default ``(1, 2, 4, 4)`` gives RF = 12 for ``tin = 12``, i.e. the stack sees the whole
    input window, and leaves exactly one temporal position to feed the head.
    """

    def __init__(self, tin, tout, channels=32, dilations=(1, 2, 4, 4), dropout=0.0):
        super().__init__()
        self.tin = int(tin)
        self.tout = int(tout)
        self.channels = int(channels)
        self.dilations = tuple(int(d) for d in dilations)
        self.n_layers = len(self.dilations)
        if self.T_after < 1:
            raise ValueError('tin=%d too small: sum(dilations)=%d leaves T_after<1'
                             % (self.tin, sum(self.dilations)))

        self.start_conv = nn.Conv2d(1, channels, kernel_size=(1, 1))
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for dil in self.dilations:
            self.filter_convs.append(nn.Conv2d(channels, channels, (1, 2), dilation=(1, dil)))
            self.gate_convs.append(nn.Conv2d(channels, channels, (1, 2), dilation=(1, dil)))
            self.residual_convs.append(nn.Conv2d(channels, channels, (1, 1)))
            self.skip_convs.append(nn.Conv2d(channels, channels, (1, 1)))
            self.bns.append(nn.BatchNorm2d(channels))
        self.dropout = nn.Dropout(dropout)
        self.end1 = nn.Conv2d(channels, channels * 2, (1, self.T_after))
        self.end2 = nn.Conv2d(channels * 2, self.tout, (1, 1))

    @property
    def T_after(self):
        return self.tin - sum(self.dilations)

    @property
    def receptive_field(self):
        return 1 + sum(self.dilations)

    def forward(self, x, A):
        """x: (B, T_in, N) -> (B, T_out, N)."""
        x = x.permute(0, 2, 1).unsqueeze(1)   # (B, T, N) -> (B, 1, N, T)
        x = self.start_conv(x)
        skip = None
        for i in range(self.n_layers):
            residual = x
            f = torch.tanh(self.filter_convs[i](x))
            g = torch.sigmoid(self.gate_convs[i](x))
            x = f * g
            x = self.bns[i](x)
            s = self.skip_convs[i](x)
            skip = s if skip is None else s + skip[..., -s.size(-1):]
            x = self.residual_convs[i](x)
            x = x + residual[..., -x.size(-1):]
            x = self.dropout(x)
            x = spmm(A, x)                   # spatial propagation over N
        x = torch.relu(skip)
        x = self.end1(x)                     # (B, 2C, N, T_after)
        x = self.end2(x)                     # (B, T_out, N, 1)
        return x.squeeze(-1)
