"""
Copyright (c) 2024 Genera1Z
https://github.com/Genera1Z
"""
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
from ..util_datum import rgb_segment_to_index_segment, draw_segmentation_np


class ClevrTex(ptud.Dataset):
    """ClevrTex: A Texture-Rich Benchmark for Unsupervised Multi-Object Segmentation
    https://www.robots.ox.ac.uk/~vgg/data/clevrtex

    Example
    ```
    dataset = ClevrTex(
        data_file="clevrtex/train.lmdb",
        extra_keys=["segment", "depth"],
        base_dir=Path("/media/GeneralZ/Storage/Static/datasets"),
    )
    for sample in dataset:
        dataset.visualiz(
            image=sample["image"].permute(1, 2, 0).numpy(),
            segment=sample["segment"].numpy(),
            depth=sample["depth"].numpy(),
        )
    ```
    """

    def __init__(
        self,
        data_file,
        extra_keys=["segment", "depth"],
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
        - depth: shape=(h,w), uint8 | float32
        """
        if not hasattr(self, "env"):  # torch>2.6
            self.env = lmdb_open_read(self.data_file)

        with self.env.begin(write=False) as txn:
            sample0 = pkl.loads(txn.get(self.idxs[index]))
        sample1 = {}

        image0 = cv2.cvtColor(
            cv2.imdecode(
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

        if "depth" in self.extra_keys:
            depth = pt.from_numpy(cv2.imdecode(sample0["depth"], cv2.IMREAD_GRAYSCALE))
            sample1["depth"] = depth  # (h,w) uint8

        sample2 = self.transform(**sample1)

        if "segment" in self.extra_keys:
            segment2 = sample2["segment"]  # index format
            segment3 = ptnf.one_hot(segment2.long()).bool()  # mask format

            sample2["segment"] = segment3  # (h,w,s) bool

        return sample2

    def __len__(self):
        return len(self.idxs)

    @staticmethod
    def convert_dataset(
        src_dir=Path("/media/GeneralZ/Storage/Static/datasets_raw/clevrtex"),
        dst_dir=Path("clevrtex"),
    ):
        """
        Download data from https://www.robots.ox.ac.uk/~vgg/data/clevrtex
        - Original Version (for train)
            - ClevrTex (part 1, 4.7 GB)
            - ClevrTex (part 2, 4.7 GB)
            - ClevrTex (part 3, 4.7 GB)
            - ClevrTex (part 4, 4.7 GB)
            - ClevrTex (part 5, 4.7 GB)
        - Also its variant (for val)
            - ClevrTex-OOD test set (5.3 GB)

        Structure dataset as follows and run it!
        - clevrtex_full  # as training set
          - 0
            - *.png
          ...
          - 49
            - *.png
        - clevrtex_outd  # as validation set
          - 0
            - *.png
          ...
          - 9
            - *.png
        """
        dst_dir.mkdir(parents=True, exist_ok=True)

        splits = dict(
            train="clevrtex_full",
            val="clevrtex_outd",
        )

        for split, image_dn in splits.items():
            image_path = src_dir / image_dn
            scenes = list(image_path.glob("**/*.png"))
            scenes.sort()

            assert len(scenes) % 6 == 0
            total_num = len(scenes) // 6  # multiple descriptions for one scene
            assert total_num in [10000, 50000]  # outd, full

            dst_file = dst_dir / f"{split}.lmdb"
            lmdb_env = lmdb_open_write(dst_file)

            keys = []
            txn = lmdb_env.begin(write=True)
            t0 = time.time()

            for cnt in range(total_num):
                files = scenes[cnt * 6 : cnt * 6 + 6]
                assert files[0].name.split(".")[0].split("_")[2].isnumeric()
                image_file = str(files[0])
                assert files[3].name.endswith("_flat.png")
                segment_file = str(files[3])
                assert files[2].name.endswith("_depth_0001.png")
                depth_file = str(files[2])

                with open(image_file, "rb") as f:
                    image_b = f.read()
                segment_bgr = cv2.imread(segment_file)  # (h,w,c=3)
                segment_rgb = cv2.cvtColor(segment_bgr, cv2.COLOR_BGR2RGB)
                segment = rgb_segment_to_index_segment(segment_rgb)  # (h,w)
                depth = cv2.imread(depth_file)[:, :, 0]  # (h,w,c=1) -> (h,w)

                # image = cv2.imdecode(np.frombuffer(image_b, "uint8"), cv2.IMREAD_COLOR)
                # __class__.visualiz(image=image, segment=segment, depth=depth, wait=0)

                sample_key = f"{cnt:06d}".encode("ascii")
                keys.append(sample_key)

                assert type(image_b) == bytes
                assert segment.ndim == 2 and segment.dtype == np.uint8
                assert depth.ndim == 2 and depth.dtype == np.uint8

                sample_dict = dict(
                    image=image_b,  # (h,w,c=3) bytes
                    segment=cv2.imencode(".webp", segment)[1],  # (h,w) uint8
                    depth=cv2.imencode(".webp", depth)[1],  # (h,w) uint8
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
    def visualiz(image, segment=None, depth=None, wait=0):
        """
        - image: rgb format, shape=(h,w,c=3), uint8
        - segment: mask format, shape=(h,w,s), bool
        - depth: shape=(h,w), uint8
        """
        assert image.ndim == 3 and image.shape[2] == 3 and image.dtype == np.uint8

        cv2.imshow("i", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        segment_viz = None
        if segment is not None:
            assert segment.ndim == 3 and segment.dtype == bool
            segment_viz = draw_segmentation_np(image, segment, alpha=0.75)
            cv2.imshow("s", cv2.cvtColor(segment_viz, cv2.COLOR_RGB2BGR))

        if depth is not None:
            assert depth.ndim == 2 and depth.dtype == np.uint8
            cv2.imshow("d", depth)

        cv2.waitKey(wait)
        return image, segment_viz, depth
