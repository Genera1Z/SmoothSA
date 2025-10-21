from abc import ABC, abstractmethod

import numpy as np
import torch as pt
import torch.nn as nn
import torch.nn.functional as ptnf

from ..util import DictTool
from ..util_learn import intersection_over_union, hungarian_matching


class MetricWrap(nn.Module):
    """"""

    def __init__(self, detach=False, **metrics):
        super().__init__()
        self.detach = detach
        self.metrics = metrics

    def forward(self, **pack: dict) -> dict:
        if self.detach:
            with pt.inference_mode():
                return self._forward(**pack)
        else:
            return self._forward(**pack)

    def _forward(self, **pack: dict) -> dict:
        metrics = {}
        for key, value in self.metrics.items():
            kwds = {t: DictTool.getattr(pack, s) for t, s in value["map"].items()}
            # if self.detach:  # inference_mode has done this
            #     kwds = {k: v.detach() for k, v in kwds.items()}
            if "transform" in value:
                kwds = value["transform"](**kwds)
            assert isinstance(value["metric"], Metric)
            metric = value["metric"](**kwds)  # (loss/acc, valid)
            if "weight" in value:
                metric = (metric[0] * value["weight"], metric[1])
            metrics[key] = metric
        return metrics

    # def compile(self):  # ??? compile everything ???
    #     for v in self.metrics.values():
    #         v["metric"].compile()


class Metric(ABC, nn.Module):
    """
    mean all or specified dimensions; always keep batch dimension
    """

    def __init__(self, mean=()):
        """
        - mean: only support ``mean``; other operations like ``sum`` are seldomly used.
            - None: do nothing
            - len(mean) == 0: mean all non-batch/first dimensions
            - len(mean) > 0: mean the specified dimensions
        """
        super().__init__()
        assert mean is None or isinstance(mean, (list, tuple))
        if isinstance(mean, (list, tuple)) and len(mean) > 0:
            assert 0 not in mean  # batch/first dimension should not be included
        self.mean = mean

    @abstractmethod
    def forward(self, *args, **kwds) -> tuple:
        ...
        metric, valid = self.finaliz(...)
        return metric, valid  # loss/acc (b,..), valid (b,)

    def finaliz(self, metric, valid=None) -> tuple:
        """mean ``metric`` along dimensions ``self.mean``; flag ``valid`` samples

        - metric: shape=(b,..); dtype=float
        - valid: shape=(b,); dtype=bool
        """
        if self.mean is None:
            metric2 = metric  # (b,..)
        elif len(self.mean) == 0:
            if metric.ndim > 1:
                metric2 = metric.flatten(1).mean(1)  # (b,)
            else:
                metric2 = metric
        else:
            metric2 = metric.mean(self.mean)  # (b,..)
        if valid is not None:
            valid2 = valid
        else:
            valid2 = pt.ones(metric.size(0), dtype=pt.bool, device=metric.device)
        return metric2, valid2


####


class CrossEntropyLoss(Metric):
    """``nn.CrossEntropyLoss``."""

    def forward(self, input, target):
        """
        - input: shape=(b,c,..), dtype=float
        - target: shape=(b,..), dtype=int64;
            or shape=(b,c,..), dtype=float
        """
        # loss = ptnf.cross_entropy(input, target, reduction="none")  # (b,..)
        loss = ptnf.cross_entropy(input, target)[None]  # (b=1,)
        return self.finaliz(loss)  # (b,) (b,)


class L1Loss(Metric):
    """``nn.L1Loss``."""

    def forward(self, input, target=None):
        if target is None:
            target = pt.zeros_like(input)
        assert input.ndim == target.ndim >= 1
        # loss = ptnf.l1_loss(input, target, reduction="none")  # (b,..)
        loss = ptnf.l1_loss(input, target)[None]  # (b=1,)
        return self.finaliz(loss)  # (b,) (b,)


class MSELoss(Metric):
    """``nn.MSELoss``."""

    def forward(self, input, target):
        assert input.ndim == target.ndim >= 1
        # TODO XXX why outside-mean is no better than builtin-mean ??? TODO XXX
        # loss = ptnf.mse_loss(input, target, reduction="none")  # (b,..)
        loss = ptnf.mse_loss(input, target)[None]  # (b=1,)
        return self.finaliz(loss)  # (b,) (b,)


####


class ClassAccuracy(Metric):
    """Classification accuracy."""

    def __init__(self, topk: int, dim=-1, mean=()):
        super().__init__(mean)
        self.topk = topk
        self.dim = dim  # the dim index of classes

    def forward(self, input, target):
        """
        input: shape=(b,..,c), dtype=int64
        target: shape=(b,..), dtype=int64
        """
        if self.topk == 1:
            predict = input.argmax(self.dim)
            assert predict.shape == target.shape
            correct = (predict == target).float()  # (b,..)
        else:
            predict = input.topk(self.topk, self.dim)[1]
            correct = (predict == target.unsqueeze(self.dim)).any(self.dim).float()
        return self.finaliz(correct)  # (b,) (b,)


class TensorSize(Metric):
    """For counting number of samples."""

    def __init__(self, dim, mean=()):
        super().__init__(mean)
        self.dim = dim

    def forward(self, input):
        size = input.shape[self.dim]
        size = pt.ones(1, device=input.device) * size
        return self.finaliz(size)


class BoxIoU(Metric):
    """
    No Hungarian matching, so not using ``mIoU`` as name.
    """

    def forward(self, input, target):
        """
        - input: shape=(b,c=4), dtype=float
        - target: shape=(b,c=4), dtype=float
        """
        assert input.shape == target.shape and input.ndim == 2 and input.size(1) == 4
        iou, valid = __class__.bbox_iou(input, target)  # (b,) (b,)
        return self.finaliz(iou, valid)  # (b,) (b,)

    @staticmethod
    def bbox_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        # Determine intersection coordinates
        x1 = pt.maximum(box1[:, 0], box2[:, 0])
        y1 = pt.maximum(box1[:, 1], box2[:, 1])
        x2 = pt.minimum(box1[:, 2], box2[:, 2])
        y2 = pt.minimum(box1[:, 3], box2[:, 3])

        # Calculate intersection area
        zero = pt.zeros_like(x1)
        intersection_area = pt.maximum(zero, x2 - x1) * pt.maximum(zero, y2 - y1)

        # Calculate areas of each box
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

        # Calculate union area
        union_area = area1 + area2 - intersection_area

        # Avoid division by zero
        box_iou = intersection_area / union_area
        valid = union_area > 0
        return box_iou, valid


class ARI(Metric):
    """"""

    def __init__(self, skip=[], mean=()):
        super().__init__(mean)
        self.skip = pt.from_numpy(np.array(skip, "int64"))

    def forward(self, input, target):
        """
        - input: shape=(b,n,c), onehot segment
        - target: shape=(b,n,d), onehot segment
        """
        assert input.ndim == target.ndim == 3
        if self.skip.numel():
            self.skip = self.skip.cuda()
            target = __class__.skip_segment(target, self.skip)
        ari = __class__.adjusted_rand_index(input, target)  # (b,)
        valid = ARI.find_valid(target)  # (b,)
        return self.finaliz(ari, valid)  # (b,) (b,)

    @pt.inference_mode()
    @staticmethod
    def adjusted_rand_index(oh_pd, oh_gt):
        """
        https://github.com/google-research/slot-attention-video/blob/main/savi/lib/metrics.py

        oh_pd: shape=(b,n,c), dtype=bool, onehot segment
        oh_gt: shape=(b,n,d), dtype=bool, onehot segment
        return: shape=(b,), dtype=float32
        """
        # the following boolean op is even slower than floating point op
        # N = (oh_gt[:, :, :, None] & oh_pd[:, :, None, :]).sum(1)  # (b,d,c)  # oom
        # long, will be auto-cast to float  # int overflow
        N = pt.einsum("bnc,bnd->bcd", oh_pd.double(), oh_gt.double()).long()  # (b,c,d)
        # the following fixed point op shows negligible speedup vesus floating point op
        A = N.sum(1)  # (b,d)
        B = N.sum(2)  # (b,c)
        num = A.sum(1)  # (b,)

        ridx = (N * (N - 1)).sum([1, 2])  # (b,)
        aidx = (A * (A - 1)).sum(1)  # (b,)
        bidx = (B * (B - 1)).sum(1)  # (b,)

        expect_ridx = (aidx * bidx) / (num * (num - 1)).clip(1)
        max_ridx = (aidx + bidx) / 2
        denominat = max_ridx - expect_ridx
        ari = (ridx - expect_ridx) / denominat  # (b,)

        # two cases ``denominat == 0``
        # - both pd and gt assign all pixels to a single cluster
        #    max_ridx == expect_ridx == ridx == num * (num - 1)
        # - both pd and gt assign max 1 point to each cluster
        #    max_ridx == expect_ridx == ridx == 0
        # we want the ARI score to be 1.0
        ari[denominat == 0] = 1
        return ari

    @pt.inference_mode()
    @staticmethod
    def skip_segment(oh_gt, skip_idx):
        """
        - oh_gt: shape=(b,n,d), onehot segment
        - skip_idx: shape=(d,), dtype=long
        """
        b, n, d = oh_gt.shape
        arange = pt.arange(d, dtype=skip_idx.dtype, device=oh_gt.device)
        mask = ~pt.isin(arange, skip_idx)
        return oh_gt[:, :, mask]

    @pt.inference_mode()
    @staticmethod
    def find_valid(oh_gt):
        assert oh_gt.ndim == 3
        valid = oh_gt.flatten(1).any(1)
        return valid


class mBO(Metric):
    """"""

    def __init__(self, skip=[], mean=()):
        super().__init__(mean)
        self.skip = pt.from_numpy(np.array(skip, "int64"))

    def forward(self, input, target):
        """
        - input: shape=(b,n,c), onehot segment
        - target: shape=(b,n,d), onehot segment
        """
        assert input.ndim == target.ndim == 3
        if self.skip.numel():
            self.skip = self.skip.cuda()
            target = ARI.skip_segment(target, self.skip)
        mbo = __class__.mean_best_overlap(input, target)  # (b,)
        valid = ARI.find_valid(target)  # (b,)
        return self.finaliz(mbo, valid)  # (b,) (b,)

    @pt.inference_mode()
    @staticmethod
    def mean_best_overlap(oh_pd, oh_gt):
        """
        https://github.com/martius-lab/videosaur/blob/main/videosaur/metrics.py

        oh_pd: shape=(b,n,c), dtype=bool, onehot segment
        oh_gt: shape=(b,n,d), dtype=bool, onehot segment
        return: shape=(b,), dtype=float32
        """
        iou_all = intersection_over_union(oh_pd, oh_gt)  # (b,c,d)
        iou, idx = iou_all.max(1)  # (b,d)
        num_gt = oh_gt.any(1).sum(1)  # (b,)
        return iou.sum(1) / num_gt  # (b,)


class mIoU(Metric):
    """"""

    def __init__(self, skip=[], mean=()):
        super().__init__(mean)
        self.skip = pt.from_numpy(np.array(skip, "int64"))

    def forward(self, input, target):
        """
        input: shape=(b,n,c), onehot segment
        target: shape=(b,n,d), onehot segment
        """
        assert input.ndim == target.ndim == 3
        if self.skip.numel():
            self.skip = self.skip.cuda()
            target = ARI.skip_segment(target, self.skip)
        miou = __class__.mean_intersection_over_union(input, target)  # (b,)
        valid = ARI.find_valid(target)  # (b,)
        return self.finaliz(miou, valid)  # (b,) (b,)

    @pt.inference_mode()
    @staticmethod
    def mean_intersection_over_union(oh_pd, oh_gt):
        """
        https://github.com/martius-lab/videosaur/blob/main/videosaur/metrics.py

        oh_pd: shape=(b,n,c), dtype=bool, onehot segment
        oh_gt: shape=(b,n,d), dtype=bool, onehot segment
        return: shape=(b,), dtype=float32
        """
        iou_all = intersection_over_union(oh_pd, oh_gt)  # (b,c,d)
        iou, idx = hungarian_matching(iou_all, maximize=True)  # (b,d)
        num_gt = oh_gt.any(1).sum(1)  # (b,)
        return iou.sum(1) / num_gt  # (b,)
