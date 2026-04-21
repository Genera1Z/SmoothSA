"""
Copyright (c) 2024 Genera1Z
https://github.com/Genera1Z
"""

from collections import defaultdict
from pathlib import Path
import json
import pickle as pkl

from einops import rearrange
from pycocotools import mask as maskUtils
from tqdm import tqdm
import cv2
import numpy as np
import torch as pt
import torch.nn.functional as ptnf
import torch.utils.data as ptud

from .dataset import lmdb_open_read, lmdb_open_write
from ..util import concurrent_pool
from ..util_datum import mask_segment_to_bbox_np, draw_segmentation_np


class YTVIS(ptud.Dataset):
    """(High-Quality) Youtube Video Instance Segmentation.
    https://arxiv.org/abs/2207.14012
    https://github.com/SysCV/vmt

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
        transform0=lambda **_: _,  # for t-slice only
        transform=lambda **_: _,
        base_dir: Path = None,
        ts=None,  # average=30.x; repeat long videos for balanced training
    ):
        if base_dir:
            data_file = base_dir / data_file
        self.data_file = data_file

        env = lmdb_open_read(data_file)
        with env.begin(write=False) as txn:
            self_keys = pkl.loads(txn.get(b"__keys__"))
        print(len(self_keys))

        if ts is None:
            self.keys = self_keys
        else:
            import time

            self.keys = []
            print(f"[{__class__.__name__}] slicing samples to length <={ts}...")
            t0 = time.time()
            for key in self_keys:
                with env.begin(write=False) as txn:
                    sample = pkl.loads(txn.get(key))
                tv = len(sample["video"])
                if tv > ts:  # sample longer videos more
                    keys = [key] * (tv // ts)
                    self.keys.extend(keys)
                else:
                    self.keys.append(key)
            print(len(self.keys))
            print(f"[{__class__.__name__}] {time.time() - t0}")

        env.close()

        self.extra_keys = extra_keys
        self.transform0 = transform0
        self.transform = transform

    def __getitem__(self, index):
        """
        - video: (t=?,c=3,h,w), uint8 | float32
        - segment: (t,h,w,s), uint8 -> bool
        - bbox: (t,s,c=4), float32. both side normalized ltrb, only foreground
        - clazz: (t,s), uint8. only foreground
        """
        if not hasattr(self, "env"):  # torch>2.6
            self.env = lmdb_open_read(self.data_file)

        with self.env.begin(write=False) as txn:
            sample0 = pkl.loads(txn.get(self.keys[index]))
        sample0 = self.transform0(**sample0)  # clip videos in advance for efficiency
        sample1 = {}

        video0 = np.array(
            [
                cv2.cvtColor(
                    cv2.imdecode(np.frombuffer(_, "uint8"), cv2.IMREAD_UNCHANGED),
                    cv2.COLOR_BGR2RGB,
                )
                for _ in sample0["video"]
            ]
        )
        video = pt.from_numpy(video0).permute(0, 3, 1, 2)
        sample1["video"] = video  # (t,c,h,w) uint8

        if "segment" in self.extra_keys:
            segment0 = np.array(
                [cv2.imdecode(_, cv2.IMREAD_UNCHANGED) for _ in sample0["segment"]]
            )
            segment = pt.from_numpy(segment0)
            sample1["segment"] = segment  # (t,h,w) uint8
            s0 = 1 + sample0["s"]  # bg+fg

            if "clazz" in self.extra_keys:
                clazz = pt.from_numpy(sample0["clazz"])
                sample1["clazz"] = clazz  # (t,s) uint8

        sample2 = self.transform(**sample1)

        if "segment" in self.extra_keys:
            segment2 = sample2["segment"]  # (t,h,w); index format
            # (t,h,w,s); mask format
            segment2_ = ptnf.one_hot(segment2.long(), s0).bool()

            t, h, w, _ = segment2_.shape

            # ``RandomCrop`` and ``CenterCrop`` can diminish segments
            cond = segment2_.any([0, 1, 2])  # (s,)
            segment3 = segment2_[:, :, :, cond]
            sample2["segment"] = segment3  # (t,h,w,s) bool

            if "bbox" in self.extra_keys:
                segment3_ = rearrange(  # skip bg
                    segment3[:, :, :, 1 if cond[0] else 0 :], "t h w s -> h w (t s)"
                )
                bbox2_ = pt.from_numpy(  # (t*s,c=4)
                    mask_segment_to_bbox_np(segment3_.numpy())
                ).float()
                bbox2 = rearrange(bbox2_, "(t s) c -> t s c", t=t)
                bbox2[:, :, 0::2] /= w  # normalize
                bbox2[:, :, 1::2] /= h
                sample2["bbox"] = bbox2  # (t,s,c=4) float32

            if "clazz" in self.extra_keys:
                clazz2 = sample2["clazz"]
                clazz2 = clazz2[:, cond[1:]]  # skip bg
                sample2["clazz"] = clazz2  # (s,) uint8

        return sample2

    def __len__(self):
        return len(self.keys)

    @staticmethod
    def convert_dataset(
        src_dir=Path("/media/GeneralZ/Storage/Static/datasets_raw/ytvis2022"),
        dst_dir=Path("ytvis_2022"),
        hq=False,  # whether using hq-ytvis or ytvis
    ):
        """
        Convert the original images into LMDB files.

        The code is adapted from
        https://github.com/SysCV/vmt/blob/main/cocoapi_hq/PythonAPI/pycocotools/ytvos.py

        ### <<< for ytvis

        Download the original dataset from
        https://youtube-vos.org/dataset/vis
            -> "Data Download" -> "2022 version" -> "Sign In" -> "Participate" -> "Get Data"
                -> "Google Drive"
        - train.zip
        - valid.zip
        - validation_gt.zip
        - test.zip
        - test_gt.json

        Unzip zip files and ensure all these files in the following structure:
        - train
            - JPEGImages  # 2985
                - 0a2f2bd294
                - 0a7a2514aa
                ...
            - instances.json
        - valid
            - JPEGImages  # 492
            - instances.json
            - gt.json
        - test
            - JPEGImages  # 569 (Official docs say it is 542?)
            - instances.json
            - test_gt.json

                Download the original dataset from
        https://youtube-vos.org/dataset/vis
            -> "Data Download" -> "2019 version new" -> "Sign In" -> "Participate" -> "Get Data"
                -> "Image frames:" -> "Baidu Pan (Passcode: uu4q)" -> "vos" -> "all_frames"
        - train_all_frames.zip  # This is split into HQ-YTVIS train/val/test
        - val_all_frames.zip    # can be skipped
        - test_all_frames.zip   # can be skipped

        ### >>>

        ### <<< for hq-ytvis

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

        ### >>>

        Finally execute this function.
        """
        dst_dir.mkdir(parents=True, exist_ok=True)
        if not hq:  # ytvis
            splits = dict(
                train=["train", "JPEGImages", "instances.json"],
                val=["valid", "JPEGImages", "gt.json"],
                test=["test", "JPEGImages", "test_gt.json"],
            )
        else:
            splits = dict(  # hq-ytvis
                train="ytvis_hq-train.json",
                val="ytvis_hq-val.json",
                test="ytvis_hq-test.json",
            )
            video_fold = src_dir / "JPEGImages"

        for split, annot_fn in splits.items():
            if not hq:
                [split_dn, video_dn, annot_fn] = annot_fn
                video_fold = src_dir / split_dn / video_dn
                print(video_fold)
                annot_file = src_dir / split_dn / annot_fn
            else:
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
            lmdb_env = lmdb_open_write(lmdb_file)

            keys = []
            txn = lmdb_env.begin(write=True)

            cnt = 0
            for vid, track_info in tqdm(track_infos.items()):
                if len(track_info) == 0:
                    continue

                frame_fns = video_infos[vid]["file_names"]  # (t,h,w,c)
                video_b = [(video_fold / _).read_bytes() for _ in frame_fns]

                t = len(track_info[0]["segmentations"])
                h = track_info[0]["height"]
                w = track_info[0]["width"]
                s = len(track_info)  # only fg

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
                    segment[mask] = j + 1  # pad bg idx
                    clazz[:, j] = clz
                assert segment.max() == s

                assert np.unique(segment).max() > 0  # have at least one object
                assert np.unique(clazz).min() > 0  # no background cls_idx

                # video = np.array(
                #     [
                #         cv2.cvtColor(
                #             cv2.imdecode(
                #                 np.frombuffer(_, "uint8"), cv2.IMREAD_UNCHANGED
                #             ),
                #             cv2.COLOR_BGR2RGB,
                #         )
                #         for _ in video_b
                #     ]
                # )
                # segment_msk = ptnf.one_hot(pt.from_numpy(segment).long()).bool().numpy()
                # __class__.visualiz(video, segment_msk, None, clazz, wait=0)

                assert segment.ndim == 3 and segment.dtype == np.uint8
                assert clazz.ndim == 2 and clazz.dtype == np.uint8

                sample_key = f"{cnt:06d}".encode("ascii")
                keys.append(sample_key)

                # v png supports dtype u8/u16, channel 1/3/4
                # x webp supports dtype u8, channel 1/3/4
                enc_param = [
                    cv2.IMWRITE_PNG_COMPRESSION,
                    9,
                    cv2.IMWRITE_PNG_STRATEGY,
                    cv2.IMWRITE_PNG_STRATEGY_FILTERED,
                ]
                segment_b = concurrent_pool(
                    lambda _: cv2.imencode(".png", _, enc_param)[1], [segment]
                )
                sample_dict = dict(
                    video=video_b,  # t*(h,w,c=3) bytes
                    segment=segment_b,  # t*(h,w) bytes
                    s=s,
                    clazz=clazz,  # (t,s) uint8
                )
                txn.put(sample_key, pkl.dumps(sample_dict))

                if (cnt + 1) % 64 == 0:  # write_freq
                    print(f"{cnt + 1:06d}")
                    txn.commit()
                    txn = lmdb_env.begin(write=True)

                cnt += 1

            txn.commit()
            txn = lmdb_env.begin(write=True)
            txn.put(b"__keys__", pkl.dumps(keys))
            txn.commit()
            lmdb_env.close()

    @staticmethod
    def convert_dataset_hq(
        src_dir=Path("/media/GeneralZ/Storage/Static/datasets_raw/ytvis_hq"),
        dst_dir=Path("ytvis_hq"),
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
            lmdb_env = lmdb_open_write(lmdb_file)

            keys = []
            txn = lmdb_env.begin(write=True)

            cnt = 0
            for vid, track_info in tqdm(track_infos.items()):
                if len(track_info) == 0:
                    continue

                frame_fns = video_infos[vid]["file_names"]  # (t,h,w,c)
                video_b = [(video_fold / _).read_bytes() for _ in frame_fns]

                t = len(track_info[0]["segmentations"])
                h = track_info[0]["height"]
                w = track_info[0]["width"]
                s = len(track_info)  # only fg

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
                    segment[mask] = j + 1  # pad bg idx
                    clazz[:, j] = clz
                assert segment.max() == s

                assert np.unique(segment).max() > 0  # have at least one object
                assert np.unique(clazz).min() > 0  # no background cls_idx

                # video = np.array(
                #     [
                #         cv2.cvtColor(
                #             cv2.imdecode(
                #                 np.frombuffer(_, "uint8"), cv2.IMREAD_UNCHANGED
                #             ),
                #             cv2.COLOR_BGR2RGB,
                #         )
                #         for _ in video_b
                #     ]
                # )
                # segment_msk = ptnf.one_hot(pt.from_numpy(segment).long()).bool().numpy()
                # __class__.visualiz(video, segment_msk, None, clazz, wait=0)

                assert segment.ndim == 3 and segment.dtype == np.uint8
                assert clazz.ndim == 2 and clazz.dtype == np.uint8

                sample_key = f"{cnt:06d}".encode("ascii")
                keys.append(sample_key)

                # v png supports dtype u8/u16, channel 1/3/4
                # x webp supports dtype u8, channel 1/3/4
                enc_param = [
                    cv2.IMWRITE_PNG_COMPRESSION,
                    9,
                    cv2.IMWRITE_PNG_STRATEGY,
                    cv2.IMWRITE_PNG_STRATEGY_FILTERED,
                ]
                segment_b = concurrent_pool(
                    lambda _: cv2.imencode(".png", _, enc_param)[1], [segment]
                )
                sample_dict = dict(
                    video=video_b,  # t*(h,w,c=3) bytes
                    segment=segment_b,  # t*(h,w) bytes
                    s=s,
                    clazz=clazz,  # (t,s) uint8
                )
                txn.put(sample_key, pkl.dumps(sample_dict))

                if (cnt + 1) % 64 == 0:  # write_freq
                    print(f"{cnt + 1:06d}")
                    txn.commit()
                    txn = lmdb_env.begin(write=True)

                cnt += 1

            txn.commit()
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
                elif type(brle["counts"]) == list:  # uncompress RLE ******always******
                    rle = maskUtils.frPyObjects(brle, h, w)
                else:  # ???
                    rle = brle
                mask = maskUtils.decode(rle)
            masks.append(mask)
        return np.array(masks)  # (t,h,w)

    @staticmethod
    def visualiz(video, segment=None, bbox=None, clazz=None, wait=0):
        """
        - video: (t,h,w,c=3) uint8, rgb format
        - segment: (t,h,w,s) bool, mask format
        - bbox: (t,s,c=4) float32, ltrb format, dual normalized
        - clazz: shape=(t,s), uint8
        """
        t, h, w, cv = video.shape
        assert cv == 3 and video.dtype == np.uint8

        if segment is not None:
            t, h, w, cs = segment.shape
            assert segment.dtype == bool

            if bbox is not None:
                t, s, cb = bbox.shape
                assert cb == 4 and bbox.dtype == np.float32
                if segment is not None:
                    assert cs - s in [0, 1]
                if clazz is not None:
                    assert clazz.shape[:2] == bbox.shape[:2]
                bbox = (bbox.copy() * [w, h, w, h]).round().astype(int)

            if clazz is not None:
                t, s = clazz.shape
                assert clazz.dtype == np.uint8
                if segment is not None:
                    assert cs - s in [0, 1]
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

                if bbox is not None and bbox.shape[0]:
                    for box in bbox[ti, :, :]:
                        seg_i_viz = cv2.rectangle(seg_i_viz, box[:2], box[2:], color=c1)

                if clazz is not None and clazz.shape[0]:
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
