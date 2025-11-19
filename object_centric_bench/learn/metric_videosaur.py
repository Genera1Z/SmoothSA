"""
Copyright (c) 2024 Genera1Z
https://github.com/Genera1Z
"""
from einops import rearrange, repeat
import torch as pt
import torch.nn.functional as ptnf

from .metric import Metric


class SlotContrastLoss(Metric):
    """Temporally Consistent Object-Centric Learning by Contrasting Slots"""

    def __init__(self, tau=0.1, mean=()):
        super().__init__(mean)
        self.tau = tau

    def forward(self, input, shift=1):
        """
        - input: slots, shape=(b,t,s,c)
        """
        b, t, s, c = input.shape
        dtype = input.dtype
        device = input.device

        slots = ptnf.normalize(input, p=2.0, dim=-1)
        slots = rearrange(slots, "b t s c -> t (s b) c")

        s1 = slots[:-shift, :, :]
        s2 = slots[shift:, :, :]
        ss = pt.einsum("tmc,tnc->tmn", s1, s2) / self.tau  # (t,s*b,s*b)
        eye = pt.eye(s * b, dtype=dtype, device=device)
        eye = repeat(eye, "m n -> t m n", t=t - shift)
        # loss = ptnf.cross_entropy(ss, eye, reduction="none")  # (b,..)
        loss = ptnf.cross_entropy(ss, eye)[None]  # (b=1,)
        return self.finaliz(loss)  # (b,..) (b,)
