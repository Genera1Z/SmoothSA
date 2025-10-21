from einops import rearrange
import numpy as np
import torch as pt
import torch.nn as nn

from ..util_learn import hungarian_matching, intersection_over_union


class ObjDiscovRecogn(nn.Module):
    """
    Supports both image and video inputs.
    """

    def __init__(
        self,
        discov: nn.Module,
        recogn: nn.Module,
        slotz_idx: int,
        attpd_idx: int,
        segpd_func,  # lambda _: interpolat_argmax_attent(_, size=resolut0).long()
        slotz_rearr: str,  # rearrange pattern; only split no merge
        segpd_rearr: str,
        seggt_rearr: str,
        clsgt_rearr: str,
        boxgt_rearr: str,
        ncls: int,
        cbox: int,
        thresh_iou=1e-2,
    ):
        super().__init__()
        self.discov = discov
        self.recogn = recogn
        self.slotz_idx = slotz_idx
        self.attpd_idx = attpd_idx
        self.segpd_func = segpd_func
        self.slotz_rearr = slotz_rearr
        self.segpd_rearr = segpd_rearr
        self.seggt_rearr = seggt_rearr
        self.clsgt_rearr = clsgt_rearr
        self.boxgt_rearr = boxgt_rearr
        self.ncls = ncls
        self.cbox = cbox
        assert thresh_iou > 0
        self.thresh_iou = thresh_iou

    def forward(self, input, condit=None, seggt=..., clsgt=..., boxgt=...):
        with pt.inference_mode(True):
            output = self.discov(input, condit)

            slotz = output[self.slotz_idx]  # (b,s,c) (b,t,s,c)
            attpd = output[self.attpd_idx]  # (b,s,h,w) (b,t,s,h,w)
            segpd = self.segpd_func(attpd)  # (b,h,w,s) (b,t,h,w,s)

        slotz = rearrange(slotz, self.slotz_rearr)
        segpd = rearrange(segpd, self.segpd_rearr)
        seggt = rearrange(seggt, self.seggt_rearr)
        clsgt = rearrange(clsgt, self.clsgt_rearr)
        boxgt = rearrange(boxgt, self.boxgt_rearr)

        slotz, clsgt, boxgt, rcidx = __class__.filter_items(
            slotz, segpd, seggt, clsgt, boxgt, self.thresh_iou
        )  # (?,c) (?,) (?,c=4) ?*(?,c=2)

        slotz = slotz.clone()  # inference mode
        clsgt = clsgt.clone()
        boxgt = boxgt.clone()

        clspd_boxpd = self.recogn(slotz)
        clspd = clspd_boxpd[:, : self.ncls]  # (?,c=ncls)
        boxpd = clspd_boxpd[:, self.ncls :]  # (?,c=4)
        assert clspd_boxpd.shape[1] == self.ncls + self.cbox
        # print(clsgt.shape, boxgt.shape)

        return slotz, clspd, boxpd, clsgt, boxgt, rcidx

    @staticmethod
    @pt.inference_mode()
    def filter_items(slotz, segpd, seggt, clsgt, boxgt, thresh_iou):
        # oh_pd = ptnf.one_hot(segpd.long())  # (b,n,s)
        # oh_gt = ptnf.one_hot(seggt.long())  # (b,n,r)
        iou = intersection_over_union(segpd, seggt[:, :, 1:])  # (b,s,r)
        iou_match, rcidx = hungarian_matching(iou, maximize=True)  # (b,min(s,r),c=2)
        # This also filters out segment paddings, along with corresponding bbox and clazz!
        iou_flag = iou_match > thresh_iou

        slotz2 = []
        clsgt2 = []
        boxgt2 = []
        rcidx2 = []

        b = slotz.size(0)
        for i in range(b):  # filter out padded items by gt
            # thresh_iou > 0 will enforce gt paddings to be filtered out
            rcidx_i0 = rcidx[i, :, :]  # (min(s,r),c=2)
            ci_i = rcidx_i0[:, 1]  # (min(s,r),)  # col idx
            _cigt_ = seggt[i, :].unique().cpu().numpy()  # (?,)
            j_i = pt.from_numpy(
                np.array(
                    [
                        j
                        for j, _ in enumerate(ci_i.cpu().numpy())
                        if _ in _cigt_ and iou_flag[i, j]
                    ]
                )
            ).to(dtype=rcidx.dtype, device=rcidx.device)
            assert (j_i.unique() >= 0).all()
            assert (j_i.unique() < rcidx_i0.size(0)).all()
            rcidx_i = rcidx_i0[j_i, :]  # (?,c=2)
            slotz_i = slotz[i, rcidx_i[:, 0], :]  # (?,c)
            clsgt_i = clsgt[i, rcidx_i[:, 1]]  # (?,)
            boxgt_i = boxgt[i, rcidx_i[:, 1], :]  # (?,)

            slotz2.append(slotz_i)
            clsgt2.append(clsgt_i)
            boxgt2.append(boxgt_i)
            rcidx2.append(rcidx_i)  # no concat to keep batch info

        slotz2 = pt.concat(slotz2)  # (??,c)
        clsgt2 = pt.concat(clsgt2)  # (??,)
        boxgt2 = pt.concat(boxgt2)  # (??,)

        return slotz2, clsgt2, boxgt2, rcidx2
