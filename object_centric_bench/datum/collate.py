import torch.utils.data as ptud

from .transform import PadTo1
from ..util import DictTool


class PadToMax1:
    """
    Pad a dimension of a list of tensors to the max size.
    A counterpart to ``DefaultCollate``.
    Not composable by ``Compose``.

    TODO XXX support ``keys=["bbox", "clazz"], pkeys=["bmask", "cmask"]``
    """

    def __init__(self, keys: list, dims: list, mode="right", value=0):
        assert len(keys) == len(dims)
        self.keys = keys
        self.dims = dims
        self.mode = mode
        self.value = value

    def __call__(self, samples: list) -> list:
        for key, dim in zip(self.keys, self.dims):
            inputs = [DictTool.getattr(_, key) for _ in samples]
            size = max(_.size(dim) for _ in inputs)

            for sample, input in zip(samples, inputs):
                left, right = PadTo1.calc_padding(size, input.size(dim), self.mode)
                output = PadTo1.pad1(input, dim, left, right, self.value)
                DictTool.setattr(sample, key, output)

        return ptud.default_collate(samples)
