from abc import ABC, abstractmethod
from bisect import bisect
from numbers import Number

import numpy as np
import torch as pt


class Schedule(ABC):
    """Schedule Everything as You Want.

    Supported cases:
    - model.alpha.fill_(value)
    - model.alpha = value
    - model.tau.data[...] = value
    - optim.param_groups[0]["lr"] = value
    - loss["recon_d"]["weight"] = value
    """

    @abstractmethod
    def __init__(self, assigns, step_count_key="step_count"):
        assert all("value" in _ for _ in assigns)
        self.assigns = assigns
        self.step_count_key = step_count_key
        self.sched = ...

    @pt.inference_mode()
    def __call__(self, **pack: dict) -> dict:
        step_count = pack[self.step_count_key]
        for k in pack.keys():  # extract all global values
            exec(f"{k} = pack['{k}']")
        for assign in self.assigns:
            value = self[step_count]
            exec(assign)  # ``value`` is executed in ``assign`` string
        return pack

    def __getitem__(self, idx):
        return self.sched[idx]

    def __len__(self):
        return len(self.sched)


class CbLinear(Schedule):

    def __init__(self, assigns, ntotal, vbase, vfinal=0):
        super().__init__(assigns)
        self.sched = [  # torch invokes lr_sched ntotal+1 times
            __class__.linear(_, ntotal, vbase, vfinal) for _ in range(ntotal + 1)
        ]

    @staticmethod
    def linear(n, ntotal, vbase, vfinal):
        return (vfinal - vbase) / ntotal * n + vbase


class CbCosine(Schedule):

    def __init__(self, assigns, ntotal, vbase, vfinal=0):
        super().__init__(assigns)
        self.sched = [
            __class__.cosine(_, ntotal, vbase, vfinal) for _ in range(ntotal + 1)
        ]

    @staticmethod
    def cosine(n, ntotal, vbase, vfinal):
        return 0.5 * (vbase - vfinal) * (1 + np.cos(np.pi * n / ntotal)) + vfinal


class CbLinearCosine(Schedule):

    def __init__(self, assigns, nlin, ntotal, vstart, vbase, vfinal=0):
        super().__init__(assigns)
        ncos = ntotal - nlin
        self.sched = [
            CbLinear.linear(_, nlin, vstart, vbase) for _ in range(nlin + 1)
        ] + [CbCosine.cosine(_, ncos, vbase, vfinal) for _ in range(1, ncos + 1)]


class CbSquarewave(Schedule):
    """
    e.g., points=[0,500,1000] and values=[1,0] means that value is 1 before step 500 while value is 0 after step 500
    """

    def __init__(self, assigns, points: list, values: list):
        super().__init__(assigns)
        assert len(values) + 1 == len(points)
        assert all(isinstance(_, Number) for _ in points)  # not nested
        # assert all(isinstance(_, Number) for _ in values)  # not nested
        self.sched = [
            __class__.squarewave(_, points, values) for _ in range(points[-1])
        ] + [values[-1]]

    @staticmethod
    def squarewave(n, points, values):
        return values[bisect(points, n) - 1]
