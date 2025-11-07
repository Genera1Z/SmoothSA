from pathlib import Path
import pickle as pkl
import time

import cv2
import lmdb
import numpy as np
import torch as pt
import torch.nn.functional as ptnf
import torch.utils.data as ptud

from .dataset import lmdb_open_read, lmdb_open_write
from ..util_datum import draw_segmentation_np


class PascalVOC(ptud.Dataset):
    """Visual Object Classes Challenge 2012 (VOC2012, train) + 2007 (val)
    - http://host.robots.ox.ac.uk/pascal/VOC/voc2012
    - http://host.robots.ox.ac.uk/pascal/VOC/voc2007

    Example
    ```
    dataset = PascalVOC(
        data_file="voc/train.lmdb",
        extra_keys=["segment"],
        base_dir=Path("/media/GeneralZ/Storage/Static/datasets"),
    )
    for sample in dataset:
        dataset.visualiz(
            image=sample["image"].permute(1, 2, 0).numpy(),
            segment=sample["segment"].numpy(),
        )
    ```
    """

    def __init__(
        self,
        data_file,
        extra_keys=["segment"],
        transform=lambda **_: _,
        base_dir: Path = None,
    ):
        if base_dir:
            data_file = base_dir / data_file
        self.data_file = data_file

        env = lmdb_open_read(data_file)
        with env.begin(write=False) as txn:
            self.idxs = pkl.loads(txn.get(b"__keys__"))
        env.close()

        self.extra_keys = extra_keys
        self.transform = transform

    def __getitem__(self, index):
        """
        - image: shape=(c=3,h,w), uint8 | float32
        - segment: shape=(h,w,s), uint8 -> bool
        """
        if not hasattr(self, "env"):  # torch>2.6
            self.env = lmdb_open_read(self.data_file)

        with self.env.begin(write=False) as txn:
            sample0 = pkl.loads(txn.get(self.idxs[index]))
        sample1 = {}

        image0 = cv2.cvtColor(
            cv2.imdecode(  # cvtColor will unify images to 3 channels safely
                np.frombuffer(sample0["image"], "uint8"), cv2.IMREAD_UNCHANGED
            ),
            cv2.COLOR_BGR2RGB,
        )
        image = pt.from_numpy(image0).permute(2, 0, 1)
        sample1["image"] = image  # (c,h,w) uint8

        if "segment" in self.extra_keys:
            segment = pt.from_numpy(
                cv2.imdecode(sample0["segment"], cv2.IMREAD_GRAYSCALE)
            )
            sample1["segment"] = segment  # (h,w) uint8

        sample2 = self.transform(**sample1)

        if "segment" in self.extra_keys:
            segment2 = sample2["segment"]  # (h,w); index format
            segment3 = ptnf.one_hot(segment2.long()).bool()  # mask format
            segment3 = segment3[:, :, 1:]  # remove boundary lines

            # ``RandomCrop`` and ``CenterCrop`` can diminish segments
            cond = segment3.any([0, 1])  # (s,)

            segment3 = segment3[:, :, cond]
            sample2["segment"] = segment3  # (h,w,s) bool

        return sample2

    def __len__(self):
        return len(self.idxs)

    @staticmethod
    def convert_dataset(
        src_dir=Path("/media/GeneralZ/Storage/Static/datasets_raw/pascalvoc/VOCdevkit"),
        dst_dir=Path("voc"),
    ):
        """
        Download the following files:
        - http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
        - http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
        - http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html#devkit
        - http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar

        Structure dataset as follows and run it!
        - VOC2012  # as training set
          - JPEGImages
            - *.jpg
          - SegmentationObject
            - *.png
        - VOC2007  # as validation set
          - JPEGImages
            - *.jpg
          - SegmentationObject
            - *.png
        """
        dst_dir.mkdir(parents=True, exist_ok=True)

        splits = dict(
            train=[
                "VOC2012/JPEGImages",
                "VOC2012/SegmentationObject",
            ],
            val=[
                "VOC2007/JPEGImages",
                "VOC2007/SegmentationObject",
            ],
        )

        for split, [image_dn, segment_dn] in splits.items():
            image_path = src_dir / image_dn
            segment_path = src_dir / segment_dn
            segment_files = list(segment_path.iterdir())
            segment_files.sort()

            dst_file = dst_dir / f"{split}.lmdb"
            lmdb_env = lmdb_open_write(dst_file)

            keys = []
            txn = lmdb_env.begin(write=True)
            t0 = time.time()

            for cnt, segment_file in enumerate(segment_files):
                fn, ext = segment_file.name.split(".")
                assert ext == "png"
                image_file = image_path / f"{fn}.jpg"

                with open(image_file, "rb") as f:
                    image_b = f.read()

                segment_bgr = cv2.imread(str(segment_file))  # (h,w,c=3)
                segment_rgb = cv2.cvtColor(segment_bgr, cv2.COLOR_BGR2RGB)
                segment0 = (
                    (segment_rgb * [[[256**0, 256**1, 256**2]]]).sum(2).astype("int32")
                )

                segment = np.zeros(segment0.shape, "uint8")
                sidx_invalid = 224 * 256**0 + 224 * 256**1 + 192 * 256**2
                sidxs = np.unique(segment0).tolist()
                sidxs.remove(sidx_invalid)
                sidxs.sort()
                segment[segment0 == sidx_invalid] = 0
                for si, sidx in enumerate(sidxs):
                    segment[segment0 == sidx] = si + 1

                # image = cv2.cvtColor(
                #     cv2.imdecode(np.frombuffer(image_b, "uint8"), cv2.IMREAD_COLOR),
                #     cv2.COLOR_BGR2RGB,
                # )
                # mask = ptnf.one_hot(pt.from_numpy(segment).long()).bool().numpy()
                # __class__.visualiz(image, mask, wait=0)

                sample_key = f"{cnt:06d}".encode("ascii")
                keys.append(sample_key)

                assert type(image_b) == bytes
                assert segment.ndim == 2 and segment.dtype == np.uint8

                sample_dict = dict(
                    image=image_b,  # (h,w,c=3) bytes
                    segment=cv2.imencode(".webp", segment)[1],  # (h,w) uint8
                )
                txn.put(sample_key, pkl.dumps(sample_dict))

                if (cnt + 1) % 64 == 0:  # write_freq
                    print(f"{cnt + 1:06d}")
                    txn.commit()
                    txn = lmdb_env.begin(write=True)

            txn.commit()
            txn = lmdb_env.begin(write=True)
            txn.put(b"__keys__", pkl.dumps(keys))
            txn.commit()
            lmdb_env.close()

            print(f"total={cnt + 1}, time={time.time() - t0}")

    @staticmethod
    def visualiz(image, segment=None, wait=0):
        """
        - image: rgb format, shape=(h,w,c=3), uint8
        - segment: mask format, shape=(h,w,s), bool
        """
        assert image.ndim == 3 and image.shape[2] == 3 and image.dtype == np.uint8

        cv2.imshow("i", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        segment_viz = None
        if segment is not None:
            assert segment.ndim == 3 and segment.dtype == bool
            segment_viz = draw_segmentation_np(image, segment, alpha=0.75)
            cv2.imshow("s", cv2.cvtColor(segment_viz, cv2.COLOR_RGB2BGR))

        cv2.waitKey(wait)
        return image, segment_viz
