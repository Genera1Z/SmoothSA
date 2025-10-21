import torch as pt

from ..util import DictTool


class Ltrb2Xywh:
    """
    Bounding box coordinates from left-top-right-bottom to center_x_y-width-height format.
    """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, **sample: dict) -> dict:
        for key in self.keys:
            input = DictTool.getattr(sample, key)  # bbox ltrb
            assert input.ndim == 2 and input.size(1) == 4
            output = __class__.ltrb2xywh(input)
            DictTool.setattr(sample, key, output)
        return sample

    @staticmethod
    def ltrb2xywh(ltrb):  # (n,c=4)
        assert ltrb.ndim == 2 and ltrb.size(1) == 4
        x = ltrb[:, 0::2].mean(1)  # (n,)
        y = ltrb[:, 1::2].mean(1)
        w = ltrb[:, 2] - ltrb[:, 0]
        h = ltrb[:, 3] - ltrb[:, 1]
        xywh = pt.stack([x, y, w, h], 1)  # (n,c=4)
        assert xywh.ndim == 2 and xywh.size(1) == 4
        return xywh


class Xywh2Ltrb(Ltrb2Xywh):

    def __call__(self, **sample: dict) -> dict:
        for key in self.keys:
            input = DictTool.getattr(sample, key)  # bbox ltrb
            assert input.ndim == 2 and input.size(1) == 4
            output = __class__.ltrb2xywh(input)
            DictTool.setattr(sample, key, output)
        return sample

    @staticmethod
    def xywh2ltrb(xywh):  # (n,c=4)
        assert xywh.ndim == 2 and xywh.size(1) == 4
        w2 = xywh[:, 2] / 2  # (n,)
        h2 = xywh[:, 3] / 2
        l = xywh[:, 0] - w2
        t = xywh[:, 1] - h2
        r = xywh[:, 2] + w2
        b = xywh[:, 3] + h2
        ltrb = pt.stack([l, t, r, b], 1)  # (n,c=4)
        assert ltrb.ndim == 2 and ltrb.size(1) == 4
        return ltrb
