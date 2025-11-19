"""
Copyright (c) 2024 Genera1Z
https://github.com/Genera1Z
"""
import torch.nn.functional as ptnf

from .metric import Metric
from ..util_learn import intersection_over_union, hungarian_matching


class AttentMatchLoss(Metric):
    """
    SPOT's attention matching loss: Hungarian matching and cross entropy.
    """

    def forward(self, input, target):
        """
        - input, target: (b,c,..) float, attention maps
        """
        input = input.flatten(2)  # (b,c,x)
        target = target.flatten(2)  # (b,d,x)
        b, c, x = input.shape
        b, d, x = target.shape

        # match
        oh_pd = ptnf.one_hot(input.argmax(1), c)  # (b,x,c)
        oh_gt = ptnf.one_hot(target.argmax(1), d)  # (b,x,d)
        iou_all = intersection_over_union(oh_pd, oh_gt)  # (b,c,d)
        iou, rcidx = hungarian_matching(iou_all, maximize=True)  # (b,d) (b,d,2)
        assert rcidx.shape[2] == 2

        # ground-truth
        # (b,d,2) -> (b,d) -> (b,1,d) -> (b,x,d)
        _ci_ = rcidx[:, :, 1][:, None, :].expand_as(oh_gt)
        # (b,x,d) -> (b,x)
        idx_gt = oh_gt.gather(2, _ci_).argmax(2)

        # loss
        loss = ptnf.cross_entropy(input, idx_gt)[None]  # (b=1,)
        return self.finaliz(loss)  # (b,) (b,)
