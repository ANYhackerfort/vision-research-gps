from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import h5py
import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]], dtype=np.float64)


class RansacMatchVerifier:
    """
    Geometric outlier removal with RANSAC.

    Modes:
      - fundamental (default): no intrinsics needed
      - essential: needs intrinsics K

    Inputs:
      - features_h5: SuperPoint features file
      - matches_h5: LightGlue matches file (pairs/<img0>__<img1>/matches0)

    Output:
      - out_h5 with groups:
          /pairs/<img0>__<img1>/
              inlier_idx0  int32 [M]   indices into img0 keypoints
              inlier_idx1  int32 [M]   indices into img1 keypoints
              num_inliers  int32 scalar
              model        float64 [3,3] (F or E) if available
    """

    def __init__(
        self,
        features_h5: Path,
        matches_h5: Path,
        out_h5: Path,
        mode: str = "fundamental",  # "fundamental" | "essential"
        intrinsics: Optional[Intrinsics] = None,
        ransac_thresh_px: float = 1.0,
        ransac_conf: float = 0.999,
        max_iters: int = 5000,
        min_inliers: int = 30,
        overwrite: bool = False,
    ) -> None:
        self.features_h5 = Path(features_h5)
        self.matches_h5 = Path(matches_h5)
        self.out_h5 = Path(out_h5)

        self.mode = mode.lower()
        if self.mode not in ("fundamental", "essential"):
            raise ValueError("mode must be 'fundamental' or 'essential'")

        if self.mode == "essential" and intrinsics is None:
            raise ValueError("essential mode requires intrinsics")

        self.intr = intrinsics
        self.thresh = float(ransac_thresh_px)
        self.conf = float(ransac_conf)
        self.max_iters = int(max_iters)
        self.min_inliers = int(min_inliers)
        self.overwrite = overwrite

    def run(self) -> Path:
        if self.out_h5.exists() and self.overwrite:
            self.out_h5.unlink()
        self.out_h5.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(self.features_h5, "r") as feats, \
             h5py.File(self.matches_h5, "r") as matches, \
             h5py.File(self.out_h5, "a") as out:

            out.attrs["verifier"] = "RANSAC"
            out.attrs["mode"] = self.mode
            out.attrs["ransac_thresh_px"] = self.thresh
            out.attrs["ransac_confidence"] = self.conf
            out.attrs["max_iters"] = self.max_iters
            out.attrs["min_inliers"] = self.min_inliers
            out.attrs["features_h5"] = str(self.features_h5)
            out.attrs["matches_h5"] = str(self.matches_h5)

            if self.intr is not None:
                out.attrs["fx"] = self.intr.fx
                out.attrs["fy"] = self.intr.fy
                out.attrs["cx"] = self.intr.cx
                out.attrs["cy"] = self.intr.cy

            pairs_in = matches.get("pairs", None)
            if pairs_in is None:
                raise RuntimeError("matches_h5 missing group: /pairs")

            pairs_out = out.require_group("pairs")

            for pair_key in pairs_in.keys():
                if pair_key in pairs_out and not self.overwrite:
                    continue
                if pair_key in pairs_out and self.overwrite:
                    del pairs_out[pair_key]

                g = pairs_in[pair_key]
                img0 = g.attrs.get("img0", None) or pair_key.split("__")[0]
                img1 = g.attrs.get("img1", None) or pair_key.split("__")[1]

                m0 = np.array(g["matches0"], dtype=np.int32)  # [N0] -> idx in img1 or -1

                kpts0 = np.array(feats[img0]["keypoints"], dtype=np.float64)  # [N0,2] (x,y)
                kpts1 = np.array(feats[img1]["keypoints"], dtype=np.float64)  # [N1,2]

                idx0 = np.where(m0 >= 0)[0].astype(np.int32)
                if idx0.size < 8:
                    self._write_empty(pairs_out, pair_key, img0, img1)
                    continue

                idx1 = m0[idx0].astype(np.int32)

                pts0 = kpts0[idx0]  # [M,2]
                pts1 = kpts1[idx1]  # [M,2]

                model, inlier_mask = self._ransac_model(pts0, pts1)
                if inlier_mask is None:
                    self._write_empty(pairs_out, pair_key, img0, img1)
                    continue

                inlier_mask = inlier_mask.reshape(-1).astype(bool)
                in0 = idx0[inlier_mask]
                in1 = idx1[inlier_mask]

                if in0.size < self.min_inliers:
                    self._write_empty(pairs_out, pair_key, img0, img1)
                    continue

                og = pairs_out.create_group(pair_key)
                og.attrs["img0"] = str(img0)
                og.attrs["img1"] = str(img1)
                og.create_dataset("inlier_idx0", data=in0.astype(np.int32), compression="gzip")
                og.create_dataset("inlier_idx1", data=in1.astype(np.int32), compression="gzip")
                og.create_dataset("num_inliers", data=np.array([in0.size], dtype=np.int32))

                if model is not None:
                    og.create_dataset("model", data=model.astype(np.float64))

        return self.out_h5

    def _ransac_model(self, pts0: np.ndarray, pts1: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self.mode == "fundamental":
            # USAC_MAGSAC is strong if available; fallback to FM_RANSAC
            method = getattr(cv2, "USAC_MAGSAC", cv2.FM_RANSAC)
            F, mask = cv2.findFundamentalMat(
                pts0, pts1,
                method=method,
                ransacReprojThreshold=self.thresh,
                confidence=self.conf,
                maxIters=self.max_iters,
            )
            if F is None or mask is None:
                return None, None
            return F, mask

        # essential mode
        K = self.intr.K()
        method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
        E, mask = cv2.findEssentialMat(
            pts0, pts1,
            cameraMatrix=K,
            method=method,
            prob=self.conf,
            threshold=self.thresh,
            maxIters=self.max_iters,
        )
        if E is None or mask is None:
            return None, None
        return E, mask

    def _write_empty(self, pairs_out: h5py.Group, pair_key: str, img0: str, img1: str) -> None:
        og = pairs_out.create_group(pair_key)
        og.attrs["img0"] = str(img0)
        og.attrs["img1"] = str(img1)
        og.create_dataset("inlier_idx0", data=np.zeros((0,), dtype=np.int32))
        og.create_dataset("inlier_idx1", data=np.zeros((0,), dtype=np.int32))
        og.create_dataset("num_inliers", data=np.array([0], dtype=np.int32))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-h5", required=True)
    ap.add_argument("--matches-h5", required=True)
    ap.add_argument("--out-h5", required=True)
    ap.add_argument("--mode", default="fundamental", choices=["fundamental", "essential"])
    ap.add_argument("--ransac-thresh", type=float, default=1.0)
    ap.add_argument("--conf", type=float, default=0.999)
    ap.add_argument("--max-iters", type=int, default=5000)
    ap.add_argument("--min-inliers", type=int, default=30)
    ap.add_argument("--overwrite", action="store_true")

    # intrinsics only needed for essential
    ap.add_argument("--fx", type=float, default=None)
    ap.add_argument("--fy", type=float, default=None)
    ap.add_argument("--cx", type=float, default=None)
    ap.add_argument("--cy", type=float, default=None)

    args = ap.parse_args()

    intr = None
    if args.mode == "essential":
        if None in (args.fx, args.fy, args.cx, args.cy):
            raise SystemExit("Essential mode requires --fx --fy --cx --cy")
        intr = Intrinsics(args.fx, args.fy, args.cx, args.cy)

    v = RansacMatchVerifier(
        features_h5=Path(args.features_h5),
        matches_h5=Path(args.matches_h5),
        out_h5=Path(args.out_h5),
        mode=args.mode,
        intrinsics=intr,
        ransac_thresh_px=args.ransac_thresh,
        ransac_conf=args.conf,
        max_iters=args.max_iters,
        min_inliers=args.min_inliers,
        overwrite=args.overwrite,
    )
    out = v.run()
    print(f"[OK] wrote verified matches: {out}")


if __name__ == "__main__":
    main()
