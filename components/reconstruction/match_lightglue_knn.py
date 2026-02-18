from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from tqdm import tqdm

from lightglue import LightGlue


@dataclass(frozen=True)
class CamRow:
    dataset: str
    name: int
    image_name: str
    image_path: str
    E: float
    N: float
    U: float


class LightGlueMatcher:
    """
    LightGlue matching using one of three pairing modes:

    Pairing modes
    ------------
    1) mode="knn"
       Match each image to its K nearest neighbors in ENU (within dataset).
       Args: --k

    2) mode="incremental"
       Match each image to neighbors in capture order (by CamRow.name),
       within a sliding window +/- W.
       Args: --window

       This is usually the BEST for "walk around building" image sets.

    3) mode="exhaustive"
       Match all unique pairs (N*(N-1)/2). Strongest connectivity, most compute.

    Inputs
    ------
      - normalized_root/cameras.csv
      - features_h5 produced by SuperPointFeatureExtractor (superpoint_features.h5)

    Output (matches_h5)
    -------------------
      /pairs/<img0>__<img1>/
          matches0  int32 [N0]  (index into img1 keypoints, -1 if no match)
          scores0   float32 [N0]
          attrs: img0, img1

    Notes
    -----
      - Uses descriptors/keypoints from features_h5.
      - Writes pairs in canonical ordering (img0 < img1) to avoid duplicates.
    """

    def __init__(
        self,
        normalized_root: Path,
        dataset: str = "ALL",
        mode: str = "knn",              # knn | incremental | exhaustive
        k: int = 8,                      # for knn
        window: int = 6,                 # for incremental
        features_h5: Optional[Path] = None,
        out_h5: Optional[Path] = None,
        device: Optional[str] = None,
        overwrite_pairs: bool = False,
        max_pairs: Optional[int] = None,  # debug
    ) -> None:
        self.normalized_root = Path(normalized_root)
        self.dataset = dataset
        self.mode = mode.lower().strip()
        self.k = int(k)
        self.window = int(window)
        self.overwrite_pairs = overwrite_pairs
        self.max_pairs = max_pairs

        if self.mode not in {"knn", "incremental", "exhaustive"}:
            raise ValueError(f"Unknown mode={mode!r}. Use knn|incremental|exhaustive")

        self.cameras_csv = self.normalized_root / "cameras.csv"

        # Default features path: <normalized_root>/../features/<dataset>/superpoint_features.h5
        if features_h5 is None:
            feat_root = self.normalized_root.parent / "features"
            feat_dir = feat_root / (dataset if dataset.upper() != "ALL" else "ALL")
            features_h5 = feat_dir / "superpoint_features.h5"
        self.features_h5 = Path(features_h5)

        # Default output matches path: <normalized_root>/../matches/<dataset>/lightglue_matches.h5
        if out_h5 is None:
            match_root = self.normalized_root.parent / "matches"
            match_dir = match_root / (dataset if dataset.upper() != "ALL" else "ALL")
            match_dir.mkdir(parents=True, exist_ok=True)
            out_h5 = match_dir / "lightglue_matches.h5"
        self.out_h5 = Path(out_h5)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.matcher = LightGlue(features="superpoint").eval().to(self.device)

    # -------------------------
    # Public API
    # -------------------------
    def run(self) -> Path:
        cams = self._load_cameras(self.cameras_csv)
        cams = self._filter_by_dataset(cams, self.dataset)

        if not cams:
            raise RuntimeError(f"No camera rows found for dataset={self.dataset!r} in {self.cameras_csv}")

        # Only match images that exist in features_h5
        with h5py.File(self.features_h5, "r") as f:
            have = set(f.keys())
        cams = [c for c in cams if c.image_name in have]

        if not cams:
            raise RuntimeError(
                f"No images for dataset={self.dataset!r} were found inside features file: {self.features_h5}"
            )

        pairs = self._build_pairs(cams)

        if self.max_pairs is not None:
            pairs = pairs[: int(self.max_pairs)]

        self.out_h5.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(self.features_h5, "r") as feats, h5py.File(self.out_h5, "a") as out:
            out.attrs["matcher"] = "LightGlue(superpoint)"
            out.attrs["device"] = self.device
            out.attrs["dataset_selection"] = self.dataset
            out.attrs["pair_mode"] = self.mode
            out.attrs["knn_k"] = int(self.k)
            out.attrs["incremental_window"] = int(self.window)
            out.attrs["features_h5"] = str(self.features_h5)

            pairs_group = out.require_group("pairs")

            desc = self._pair_desc()
            for img0, img1 in tqdm(pairs, desc=desc):
                pair_key = f"{img0}__{img1}"
                if pair_key in pairs_group and not self.overwrite_pairs:
                    continue
                if pair_key in pairs_group and self.overwrite_pairs:
                    del pairs_group[pair_key]

                f0 = self._read_feature(feats, img0)
                f1 = self._read_feature(feats, img1)
                m0, s0 = self._match_pair(f0, f1)

                g = pairs_group.create_group(pair_key)
                g.create_dataset("matches0", data=m0.astype(np.int32), compression="gzip")
                g.create_dataset("scores0", data=s0.astype(np.float32), compression="gzip")
                g.attrs["img0"] = img0
                g.attrs["img1"] = img1

        return self.out_h5

    def _pair_desc(self) -> str:
        if self.mode == "knn":
            return f"LightGlue matching ({self.dataset}, mode=knn, k={self.k})"
        if self.mode == "incremental":
            return f"LightGlue matching ({self.dataset}, mode=incremental, window={self.window})"
        return f"LightGlue matching ({self.dataset}, mode=exhaustive)"

    # -------------------------
    # Loading + filtering
    # -------------------------
    def _load_cameras(self, path: Path) -> List[CamRow]:
        rows: List[CamRow] = []
        with path.open("r", newline="") as f:
            r = csv.DictReader(f)
            for line in r:
                rows.append(
                    CamRow(
                        dataset=str(line["dataset"]),
                        name=int(line["name"]),
                        image_name=str(line["image_name"]),
                        image_path=str(line.get("image_path", "")),
                        E=float(line["E"]),
                        N=float(line["N"]),
                        U=float(line["U"]),
                    )
                )
        return rows

    def _filter_by_dataset(self, cams: List[CamRow], dataset: str) -> List[CamRow]:
        if dataset.upper() == "ALL":
            return cams
        return [c for c in cams if c.dataset == dataset]

    # -------------------------
    # Pair building (3 modes)
    # -------------------------
    def _build_pairs(self, cams: List[CamRow]) -> List[Tuple[str, str]]:
        if self.mode == "knn":
            return self._build_knn_pairs(cams, self.k)
        if self.mode == "incremental":
            return self._build_incremental_pairs(cams, self.window)
        return self._build_exhaustive_pairs(cams)

    def _build_knn_pairs(self, cams: List[CamRow], k: int) -> List[Tuple[str, str]]:
        cams_sorted = sorted(cams, key=lambda c: (c.dataset, c.name, c.image_name))
        X = np.array([[c.E, c.N, c.U] for c in cams_sorted], dtype=np.float64)
        names = [c.image_name for c in cams_sorted]

        # Full distance matrix (ok for N up to a few thousand)
        diff = X[:, None, :] - X[None, :, :]
        D = np.sum(diff * diff, axis=2)
        np.fill_diagonal(D, np.inf)

        pairs: List[Tuple[str, str]] = []
        seen = set()

        for i in range(len(names)):
            nn = np.argsort(D[i])[:k]
            for j in nn:
                a, b = names[i], names[j]
                if a == b:
                    continue
                key = (a, b) if a < b else (b, a)
                if key in seen:
                    continue
                seen.add(key)
                pairs.append(key)

        return pairs

    def _build_incremental_pairs(self, cams: List[CamRow], window: int) -> List[Tuple[str, str]]:
        # order by capture index (name)
        cams_sorted = sorted(cams, key=lambda c: (c.dataset, c.name, c.image_name))
        names = [c.image_name for c in cams_sorted]

        pairs: List[Tuple[str, str]] = []
        seen = set()

        W = max(1, int(window))
        n = len(names)
        for i in range(n):
            a = names[i]
            lo = max(0, i - W)
            hi = min(n - 1, i + W)
            for j in range(lo, hi + 1):
                if j == i:
                    continue
                b = names[j]
                key = (a, b) if a < b else (b, a)
                if key in seen:
                    continue
                seen.add(key)
                pairs.append(key)

        return pairs

    def _build_exhaustive_pairs(self, cams: List[CamRow]) -> List[Tuple[str, str]]:
        cams_sorted = sorted(cams, key=lambda c: (c.dataset, c.name, c.image_name))
        names = [c.image_name for c in cams_sorted]

        pairs: List[Tuple[str, str]] = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                pairs.append((names[i], names[j]))  # already i<j, so canonical
        return pairs

    # -------------------------
    # Feature reading
    # -------------------------
    def _read_feature(self, feats_h5: h5py.File, image_name: str) -> Dict[str, torch.Tensor]:
        g = feats_h5[image_name]
        kpts = np.array(g["keypoints"], dtype=np.float32)      # [N,2] (x,y)
        desc = np.array(g["descriptors"], dtype=np.float32)    # [N,D]
        if "scores" in g:
            scr = np.array(g["scores"], dtype=np.float32)
        else:
            scr = np.ones((kpts.shape[0],), dtype=np.float32)

        return {
            "keypoints": torch.from_numpy(kpts)[None, ...].to(self.device),
            "descriptors": torch.from_numpy(desc)[None, ...].to(self.device),
            "keypoint_scores": torch.from_numpy(scr)[None, ...].to(self.device),
        }

    # -------------------------
    # Match one pair
    # -------------------------
    @torch.inference_mode()
    def _match_pair(self, f0: Dict[str, torch.Tensor], f1: Dict[str, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
        if self.device.startswith("cuda"):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = self.matcher({"image0": f0, "image1": f1})
        else:
            out = self.matcher({"image0": f0, "image1": f1})

        m0_t = out["matches0"][0]

        if "scores0" in out:
            s0_t = out["scores0"][0]
        elif "matching_scores0" in out:
            s0_t = out["matching_scores0"][0]
        else:
            # fallback: 1 for matched, 0 for unmatched
            m0_cpu = m0_t.detach().cpu()
            s0_t = (m0_cpu >= 0).float()

        m0 = m0_t.detach().cpu().numpy().astype(np.int32)
        s0 = s0_t.detach().cpu().numpy().astype(np.float32)
        return m0, s0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--normalized-root", required=True, help="Path to outputs/normalized")
    ap.add_argument("--dataset", default="ALL", help='Dataset/building name like "Campbell" or "Chem" or "ALL"')

    ap.add_argument("--mode", default="knn", choices=["knn", "incremental", "exhaustive"],
                    help="Pairing strategy for which image pairs to match")
    ap.add_argument("--k", type=int, default=8, help="(knn mode) K nearest neighbors (ENU) per image")
    ap.add_argument("--window", type=int, default=6, help="(incremental mode) +/- window by capture index (name)")

    ap.add_argument("--features-h5", default=None, help="Path to superpoint_features.h5 (optional)")
    ap.add_argument("--out-h5", default=None, help="Output matches h5 path (optional)")
    ap.add_argument("--device", default=None, help='Force device: "cuda" or "cpu"')
    ap.add_argument("--overwrite-pairs", action="store_true", help="Overwrite existing pairs in output h5")
    ap.add_argument("--max-pairs", type=int, default=None, help="Debug: only run first N pairs")

    args = ap.parse_args()

    m = LightGlueMatcher(
        normalized_root=Path(args.normalized_root),
        dataset=args.dataset,
        mode=args.mode,
        k=args.k,
        window=args.window,
        features_h5=Path(args.features_h5) if args.features_h5 else None,
        out_h5=Path(args.out_h5) if args.out_h5 else None,
        device=args.device,
        overwrite_pairs=args.overwrite_pairs,
        max_pairs=args.max_pairs,
    )
    out = m.run()
    print(f"[OK] wrote matches: {out}")


if __name__ == "__main__":
    main()
