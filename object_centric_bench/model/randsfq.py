"""
Reasoning-Enhanced Object-Centric Learning for Videos
https://github.com/intell-sci-comput/STATM

SlotPi: Physics-informed Object-centric Reasoning Models
https://github.com/intell-sci-comput/SlotPi

Object-Centric Video Prediction via Decoupling of Object Dynamics and Interactions
https://github.com/hanoonaR/object-centric-ovd
"""

import torch as pt
import torch.nn as nn


class RSFQTransit(nn.Module):

    def __init__(
        self, dt, ci, c, nhead=4, expanz=4, pdo=0.5, norm_first=False, bias=False
    ):
        super().__init__()
        self.dt = dt
        self.te = nn.Embedding(dt, c)
        self.proji = nn.Linear(ci, c)
        self.transit = nn.TransformerDecoderLayer(
            d_model=c,
            nhead=nhead,
            dim_feedforward=c * expanz,
            dropout=pdo,
            activation="gelu",
            batch_first=True,
            norm_first=norm_first,
            bias=bias,
        )

    def forward(self, slotzs0, encodes0):
        """
        slotzs: all past step slots, shape=(b,t=i,n,c)
        encodes: all past and current step frame features, shape=(b,i+1,h*w,c)
        """
        # slotzs = slotzs[:, -self.dt :, :, :]  # window size <=dt
        # encodes = encodes[:, -(self.dt + 1) :, :, :]  # window size <=dt+1
        # NOTE This assumes training window size == dt, otherwise there will be error in self.te(ts) !!!
        #   Thus ``slotzs.size(1) always< self.dt`` and ``encodes.size(1) always<= self.dt`` !!!
        # Below is the corrected implementation. TODO Not sure if equivalent during training.
        slotzs = slotzs0[:, -self.dt + 1 :, :, :]  # window size <=dt
        encodes = encodes0[:, -self.dt :, :, :]  # window size <=dt+1

        b, i, n, c = slotzs.shape
        assert i + 1 == encodes.size(1)
        device = slotzs.device
        bidx = pt.arange(b, dtype=pt.int64, device=device)

        if self.training and i > 1:
            ts = pt.randint(0, i, [b], dtype=pt.int64, device=device)
            slotz = slotzs[bidx, ts, :, :]
            dts = i - ts
            # ensuring ts<te: always bad
            te = pt.randint(1, i + 1, [b], dtype=pt.int64, device=device)
            encode = encodes[bidx, te, :, :]
            dte = i - te
            # print(i, dts.tolist(), ts.tolist())
            # print(i, dte.tolist(), te.tolist())
            # import pdb; pdb.set_trace()
        else:
            slotz = slotzs[:, -1, :, :]
            dts = pt.ones(b, dtype=pt.int64, device=device)
            encode = encodes[:, -1, :, :]
            dte = pt.zeros(b, dtype=pt.int64, device=device)
        # ### <<<- experiment
        # if i + 1 >= self.dt:
        #     print(self.ts_, self.te_)
        #     ts = pt.ones(b, dtype=pt.int64, device=device) * self.ts_
        #     slotz = slotzs[bidx, ts, :, :]
        #     dts = i - ts
        #     te = pt.ones(b, dtype=pt.int64, device=device) * self.te_
        #     encode = encodes[bidx, te, :, :]
        #     dte = i - te
        # else:
        #     slotz = slotzs[:, -1, :, :]
        #     dts = pt.ones(b, dtype=pt.int64, device=device)
        #     encode = encodes[:, -1, :, :]
        #     dte = pt.zeros(b, dtype=pt.int64, device=device)
        # ### ->>>

        tes = self.te(dts)[:, None, :]
        slotz = slotz.detach() + tes
        tee = self.te(dte)[:, None, :]
        encode = self.proji(encode.detach()) + tee

        query = self.transit(slotz, encode)
        return query
