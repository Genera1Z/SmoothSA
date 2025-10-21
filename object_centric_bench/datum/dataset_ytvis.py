from collections import defaultdict
from pathlib import Path
import json
import pickle as pkl
import time

from einops import rearrange
from pycocotools import mask as maskUtils
import cv2
import lmdb
import numpy as np
import torch as pt
import torch.nn.functional as ptnf
import torch.utils.data as ptud

from ..util_datum import draw_segmentation_np, mask_segment_to_bbox_np


class YTVIS(ptud.Dataset):
    """(High-Quality) Youtube Video Instance Segmentation.
    https://arxiv.org/abs/2207.14012
    https://github.com/SysCV/vmt

    Number of objects distribution
    - train: {1: 1143, 2: 902, 3: 184, 4: 68, 5: 14, 6: 8}
    - val: {1: 207, 2: 125, 3: 28, 4: 15, 5: 2}
    - test: {1: 202, 2: 146, 3: 41, 4: 13}

    Example
    ```
    dataset = YTVIS(
        data_file="ytvis/train.lmdb",
        extra_keys=["segment", "bbox", "clazz"],
        base_dir=Path("/media/GeneralZ/Storage/Static/datasets"),
    )
    for sample in dataset:
        dataset.visualiz(
            video=sample["video"].permute(0, 2, 3, 1).numpy(),
            segment=sample["segment"].numpy(),
            bbox=sample["bbox"].numpy(),
            clazz=sample["clazz"].numpy(),
        )
    ```
    """

    def __init__(
        self,
        data_file,
        extra_keys=["segment", "bbox", "clazz"],
        transform=lambda **_: _,
        max_spare=4,
        base_dir: Path = None,
        ts=20,  # least number of time steps
    ):
        if base_dir:
            data_file = base_dir / data_file
        self.env = lmdb.open(
            str(data_file),
            subdir=False,
            readonly=True,
            readahead=False,
            meminit=False,
            max_spare_txns=max_spare,
            lock=False,
        )
        with self.env.begin(write=False) as txn:
            self_keys = pkl.loads(txn.get(b"__keys__"))
        print(len(self_keys))

        self.keys = []
        print(f"[{__class__.__name__}] slicing samples in dataset...")
        t0 = time.time()
        for key in self_keys:
            with self.env.begin(write=False) as txn:
                sample = pkl.loads(txn.get(key))
            t = len(sample["video"])
            if t < ts:
                continue
            num = int(np.ceil(t / ts))  # split ts>20 into multiple
            for i in range(num):
                start = (i * ts) if (i + 1 < num) else (t - ts)
                end = start + ts
                if end > num:
                    start = end - ts
                self.keys.append([key, start])
        print(f"[{__class__.__name__}] {time.time() - t0}")

        self.extra_keys = extra_keys
        self.transform = transform
        self.ts = ts

    def __getitem__(self, index):
        """
        - video: (t=20,c=3,h,w), uint8 | float32
        - segment: (t,h,w,s), uint8 -> bool
        - bbox: (t,s,c=4), float32. both side normalized ltrb, only foreground
        - clazz: (t,s), uint8. only foreground
        """
        key, start = self.keys[index]
        with self.env.begin(write=False) as txn:
            sample0 = pkl.loads(txn.get(key))
        sample1 = {}

        video = sample0["video"][start : start + self.ts]
        video = np.array(
            [
                cv2.cvtColor(
                    cv2.imdecode(np.frombuffer(_, "uint8"), cv2.IMREAD_UNCHANGED),
                    cv2.COLOR_BGR2RGB,
                )
                for _ in video
            ]
        )
        video = pt.from_numpy(video).permute(0, 3, 1, 2)
        sample1["video"] = video  # (t,c,h,w) uint8

        if "segment" in self.extra_keys:
            segment = sample0["segment"][start : start + self.ts]
            segment = np.array([cv2.imdecode(_, cv2.IMREAD_GRAYSCALE) for _ in segment])
            segment = pt.from_numpy(segment)
            sample1["segment"] = segment  # (t,h,w) uint8

            if "clazz" in self.extra_keys:
                clazz = pt.from_numpy(sample0["clazz"][start : start + self.ts])
                sample1["clazz"] = clazz  # (t,s) uint8

        sample2 = self.transform(**sample1)

        if "segment" in self.extra_keys:
            segment2 = sample2["segment"]  # (h,w) index format
            if "clazz" in self.extra_keys:
                s0 = clazz.shape[1] + 1
                # in case current video slice has less objects than total
                segment3 = ptnf.one_hot(segment2.long(), s0).bool()
            else:
                segment3 = ptnf.one_hot(segment2.long()).bool()

            t, h, w, s = segment3.shape

            # ``RandomCrop`` and ``CenterCrop`` can diminish segments
            cond = segment3.any([0, 1, 2])  # (s,)

            segment3 = segment3[:, :, :, cond]
            sample2["segment"] = segment3  # (t,h,w,s) bool

            if "bbox" in self.extra_keys:
                segment3_ = rearrange(
                    segment3[:, :, :, 1:], "t h w s -> h w (t s)"
                )  # [:, :, :, 1:] skip bg
                bbox2_ = pt.from_numpy(  # (t*s,c=4)
                    mask_segment_to_bbox_np(segment3_.numpy())
                ).float()
                bbox2 = rearrange(bbox2_, "(t s) c -> t s c", t=t)
                bbox2[:, :, 0::2] /= w  # normalize
                bbox2[:, :, 1::2] /= h
                sample2["bbox"] = bbox2  # (t,s,c=4) float32
                # print("bbox after", sample2["bbox"].shape)

            if "clazz" in self.extra_keys:
                # print("clazz before", sample2["clazz"].shape)
                if cond.shape[0] - 1 != sample2["clazz"].shape[1]:
                    print(cond.shape[0], sample2["clazz"].shape[1])
                    print()
                clazz2 = sample2["clazz"][:, cond[1:]]  # [1:] skip bg
                sample2["clazz"] = clazz2  # (s,) uint8
                # print("clazz after", clazz2.shape)

        return sample2

    def __len__(self):
        return len(self.keys)

    @staticmethod
    def convert_dataset(
        src_dir=Path("/media/GeneralZ/Storage/Static/datasets_raw/ytvis"),
        dst_dir=Path("ytvis"),
    ):
        """
        Convert the original images into LMDB files.

        The code is adapted from
        https://github.com/SysCV/vmt/blob/main/cocoapi_hq/PythonAPI/pycocotools/ytvos.py

        Download the original dataset from
        https://youtube-vos.org/dataset/vis
            -> "Data Download" -> "2019 version new" -> "Sign In" -> "Participate" -> "Get Data"
                -> "Image frames:" -> "Baidu Pan (Passcode: uu4q)" -> "vos" -> "all_frames"
        - train_all_frames.zip  # This is split into HQ-YTVIS train/val/test
        - val_all_frames.zip    # can be skipped
        - test_all_frames.zip   # can be skipped

        Download the high-quality annotation files from
        https://github.com/SysCV/vmt
            -> "Dataset Download: HQ-YTVIS Annotation Link"
                -> https://drive.google.com/drive/folders/1ZU8_qO8HnJ_-vvxIAn8-_kJ4xtOdkefh
        - ytvis_hq-train.json
        - ytvis_hq-val.json
        - ytvis_hq-test.json

        Unzip zip files and ensure all these files in the following structure:
        - JPEGImages  # video frames of train/val/test are all here
            - 0a2f2bd294
            - 0a7a2514aa
            ...
        - ytvis_hq-train.json
        - ytvis_hq-val.json
        - ytvis_hq-test.json

        Finally execute this function.
        """
        dst_dir.mkdir(parents=True, exist_ok=True)

        splits = dict(
            train="ytvis_hq-train.json",
            val="ytvis_hq-val.json",
            test="ytvis_hq-test.json",
        )
        video_fold = src_dir / "JPEGImages"

        for split, annot_fn in splits.items():
            print(split, annot_fn)
            annot_file = src_dir / annot_fn
            with open(annot_file, "r") as fi:
                annot = json.load(fi)

            video_infos = {}
            for vinfo in annot["videos"]:
                video_infos[vinfo["id"]] = vinfo

            track_infos = defaultdict(list)
            for tinfo in annot["annotations"]:
                track_infos[tinfo["video_id"]].append(tinfo)

            lmdb_file = dst_dir / f"{split}.lmdb"
            lmdb_env = lmdb.open(
                str(lmdb_file),
                map_size=1024**4,
                subdir=False,
                readonly=False,
                meminit=False,
            )

            keys = []
            txn = lmdb_env.begin(write=True)
            t0 = time.time()

            cnt = 0
            for vid, track_info in track_infos.items():
                if len(track_info) == 0:
                    continue

                frame_fns = video_infos[vid]["file_names"]  # (t,h,w,c)
                video_b = [(video_fold / _).read_bytes() for _ in frame_fns]

                t = len(track_info[0]["segmentations"])
                h = track_info[0]["height"]
                w = track_info[0]["width"]
                s = len(track_info)

                assert all(h == _["height"] for _ in track_info)
                assert all(w == _["width"] for _ in track_info)
                assert all(t == len(_["segmentations"]) for _ in track_info)
                assert t == len(video_b)

                segment = np.zeros([t, h, w], "uint8")
                # only keep the class of foreground objects
                clazz = np.zeros([t, s], "uint8")
                for j, track in enumerate(track_info):
                    assert j + 1 < 256
                    mask = __class__.rle_to_mask(track, h, w)
                    assert set(np.unique(mask)) <= {0, 1}
                    mask = mask.astype("bool")
                    clz = track["category_id"]
                    assert clz > 0
                    segment[mask] = j + 1
                    clazz[:, j] = clz

                assert np.unique(segment).max() > 0  # have at least one object
                assert np.unique(clazz).min() > 0  # no background cls_idx

                # video = np.array(
                #     [
                #         cv2.cvtColor(
                #             cv2.imdecode(np.frombuffer(_, "uint8"), cv2.IMREAD_COLOR),
                #             cv2.COLOR_BGR2RGB,
                #         )
                #         for _ in video_b
                #     ]
                # )
                # segment_pt = pt.from_numpy(segment).long()
                # mask = ptnf.one_hot(segment_pt, s + 1).bool().numpy()
                # __class__.visualiz(video, mask, None, clazz, wait=0)

                sample_key = f"{cnt:06d}".encode("ascii")
                keys.append(sample_key)

                sample_dict = dict(
                    video=video_b,  # (t,h,w,c=3) bytes
                    segment=[  # (t,h,w) bytes
                        cv2.imencode(".webp", _)[1] for _ in segment
                    ],
                    clazz=clazz,  # (t,s) uint8
                )
                txn.put(sample_key, pkl.dumps(sample_dict))

                if (cnt + 1) % 64 == 0:  # write_freq
                    print(f"{cnt + 1:06d}")
                    txn.commit()
                    txn = lmdb_env.begin(write=True)

                cnt += 1

            txn.commit()
            print((time.time() - t0) / cnt)

            txn = lmdb_env.begin(write=True)
            txn.put(b"__keys__", pkl.dumps(keys))
            txn.commit()
            lmdb_env.close()

    @staticmethod
    def rle_to_mask(track, h, w):
        masks = []
        for frameId in range(len(track["segmentations"])):
            brle = track["segmentations"][frameId]
            if brle is None:  # not visible
                mask = np.zeros([h, w], "uint8")
            else:
                if type(brle) == list:  # polygon; merge parts belonging to one object
                    rles = maskUtils.frPyObjects(brle, h, w)
                    rle = maskUtils.merge(rles)
                elif type(brle["counts"]) == list:  # uncompress RLE
                    rle = maskUtils.frPyObjects(brle, h, w)
                else:  # ???
                    rle = brle
                mask = maskUtils.decode(rle)
            masks.append(mask)
        return np.array(masks)  # (t,h,w)

    @staticmethod
    def visualiz(video, segment=None, bbox=None, clazz=None, wait=0):
        """
        - video: (t,h,w,c=3) uint8. rgb format
        - segment: (t,h,w,s) bool. mask format
        - bbox: (t,s,c=4) float32. both side normalized ltrb
        - clazz: shape=(t,s), uint8
        """
        assert video.ndim == 4 and video.shape[3] == 3 and video.dtype == np.uint8
        t, h, w, c = video.shape

        segment_viz = None
        if segment is not None and segment.shape[3]:
            assert segment.ndim == 4 and segment.dtype == np.bool

            if bbox is not None:
                assert (
                    bbox.ndim == 3 and bbox.shape[2] == 4 and bbox.dtype == np.float32
                )
                if clazz is not None:
                    assert clazz.shape[:2] == bbox.shape[:2]

                bbox[:, :, 0::2] *= w
                bbox[:, :, 1::2] *= h
                bbox = bbox.astype("int")

            if clazz is not None:
                assert clazz.ndim == 2 and clazz.dtype == np.uint8
                if bbox is not None:
                    assert clazz.shape[:2] == bbox.shape[:2]

        c1 = (255, 255, 255)
        frames = []
        segments_viz = []

        for ti, frame in enumerate(video):
            cv2.imshow("v", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            frames.append(frame)

            if segment is not None and segment.shape[3]:
                seg_i = segment[ti, :, :, :]
                seg_i_viz = draw_segmentation_np(frame, seg_i, alpha=0.75)

                if bbox is not None:
                    for box in bbox[ti, :, :]:
                        seg_i_viz = cv2.rectangle(seg_i_viz, box[:2], box[2:], color=c1)

                if clazz is not None:
                    for ci, clz in enumerate(clazz[ti, :]):
                        msk = seg_i[:, :, ci + 1]  # skip bg
                        total = float(np.sum(msk))
                        if total == 0:
                            continue
                        ys, xs = np.indices(msk.shape)  # centroid
                        cx = int(round((xs * msk).sum() / total))
                        cy = int(round((ys * msk).sum() / total))

                        seg_i_viz = cv2.putText(
                            seg_i_viz,
                            f"{clz}",
                            [cx, cy],
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            [255] * 3,
                        )

                cv2.imshow("s", cv2.cvtColor(seg_i_viz, cv2.COLOR_RGB2BGR))
                segments_viz.append(seg_i_viz)

            cv2.waitKey(wait)

        return frames, segments_viz
