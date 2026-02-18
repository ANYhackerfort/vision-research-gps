from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import h5py
import numpy as np
import torch
import yaml
from tqdm import tqdm

from lightglue import SuperPoint
torch.backends.cudnn.benchmark = True


@dataclass(frozen=True)
class ImageItem:
    dataset: str
    image_name: str
    image_path: Path


class SuperPointFeatureExtractor:
    """
    Extract SuperPoint features for a selected dataset (building) from your normalized poses file.

    Expected inputs (from your normalization step):
      - <root>/outputs/normalized/poses_normalized.yaml
      - <root>/outputs/normalized/images/<canonical_images...>

    Outputs:
      - <root>/outputs/features/<dataset>/superpoint_features.h5   (or ALL/)
        HDF5 layout:
          /<image_name>/keypoints   float32 [N,2]  (x,y)
          /<image_name>/descriptors float32 [N,D]
          /<image_name>/scores      float32 [N]
          /<image_name>/image_size  int32   [2]    (h,w)
          /<image_name>/dataset     utf-8 str attr
    """

    def __init__(
        self,
        normalized_root: Path,
        dataset: str = "ALL",
        out_root: Optional[Path] = None,
        max_side: Optional[int] = 1600,
        max_keypoints: int = 2048,
        device: Optional[str] = None,
        overwrite: bool = False,
    ) -> None:
        self.normalized_root = Path(normalized_root)
        self.dataset = dataset
        self.max_side = max_side
        self.max_keypoints = max_keypoints
        self.overwrite = overwrite

        self.poses_yaml = self.normalized_root / "poses_normalized.yaml"
        self.images_dir = self.normalized_root / "images"

        self.out_root = Path(out_root) if out_root else (self.normalized_root.parent / "features")
        self.out_dir = self.out_root / (dataset if dataset.upper() != "ALL" else "ALL")
        self.out_dir.mkdir(parents=True, exist_ok=True)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = SuperPoint(max_num_keypoints=self.max_keypoints).eval().to(self.device)

    # -----------------------------
    # Public API
    # -----------------------------
    def run(self) -> Path:
        scenes = self._load_scenes(self.poses_yaml)
        items = self._collect_items(scenes, self.dataset)

        out_h5 = self.out_dir / "superpoint_features.h5"
        if out_h5.exists() and self.overwrite:
            out_h5.unlink()

        mode = "a"  # append/update
        with h5py.File(out_h5, mode) as h5:
            h5.attrs["extractor"] = "SuperPoint (LightGlue)"
            h5.attrs["device"] = self.device
            h5.attrs["max_side"] = -1 if self.max_side is None else int(self.max_side)
            h5.attrs["max_keypoints"] = int(self.max_keypoints)
            h5.attrs["dataset_selection"] = self.dataset

            for it in tqdm(items, desc=f"SuperPoint ({self.dataset})"):
                gname = it.image_name
                if gname in h5 and (not self.overwrite):
                    continue
                if gname in h5 and self.overwrite:
                    del h5[gname]

                img = self._read_gray(it.image_path, self.max_side)
                feats = self._extract_one(img)

                grp = h5.create_group(gname)
                grp.create_dataset("keypoints", data=feats["keypoints"], compression="gzip")
                grp.create_dataset("descriptors", data=feats["descriptors"], compression="gzip")
                grp.create_dataset("scores", data=feats["scores"], compression="gzip")
                grp.create_dataset("image_size", data=np.array(img.shape[:2], dtype=np.int32))
                grp.attrs["dataset"] = it.dataset
                grp.attrs["source_path"] = str(it.image_path)

        return out_h5

    # -----------------------------
    # YAML loading + filtering
    # -----------------------------
    def _load_scenes(self, poses_yaml: Path) -> List[Dict]:
        doc = yaml.safe_load(poses_yaml.read_text()) or {}
        scenes = doc.get("scenes", [])
        if not scenes:
            raise RuntimeError(f"No scenes found in {poses_yaml}")
        return scenes

    def _collect_items(self, scenes: List[Dict], dataset: str) -> List[ImageItem]:
        """
        Build list of ImageItem using canonical normalized images directory first:
        normalized_root/images/<image_name>
        Fallback to the YAML image_path (absolute or relative) only if the canonical file is missing.
        """
        want_all = dataset.upper() == "ALL"
        items: List[ImageItem] = []

        for scene in scenes:
            for p in (scene.get("poses", []) or []):
                ds = str(p.get("dataset", "") or "")
                if not (want_all or ds == dataset):
                    continue

                img_name = str(p.get("image_name", "") or "")
                if not img_name:
                    # skip if no canonical name present
                    continue

                # 1) Prefer canonical normalized image path
                canonical = (self.images_dir / img_name)
                canonical = canonical.resolve()

                if canonical.exists():
                    img_path = canonical
                else:
                    # 2) fallback: try YAML-provided image_path (absolute or relative)
                    raw = str(p.get("image_path", "") or "")
                    if not raw:
                        # no alternative; still append canonical (will fail later with clear message)
                        img_path = canonical
                    else:
                        ypath = Path(raw).expanduser()
                        if ypath.is_absolute():
                            img_path = ypath.resolve()
                        else:
                            # resolve relative to the directory containing poses_normalized.yaml
                            # which is self.normalized_root (expected)
                            img_path = (self.normalized_root / ypath).resolve()

                items.append(ImageItem(dataset=ds, image_name=img_name, image_path=img_path))

        if not items:
            raise RuntimeError(f"No images found for dataset={dataset}")
        return items

    # -----------------------------
    # Image IO
    # -----------------------------
    def _read_gray(self, path: Path, max_side: Optional[int]) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")

        if max_side is not None:
            h, w = img.shape[:2]
            m = max(h, w)
            if m > max_side:
                scale = max_side / float(m)
                nh, nw = int(round(h * scale)), int(round(w * scale))
                img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

        return img

    # -----------------------------
    # SuperPoint extraction
    # -----------------------------
    @torch.inference_mode()
    def _extract_one(self, gray: np.ndarray) -> Dict[str, np.ndarray]:
        # to tensor: [1,1,H,W] float in [0,1]
        t = torch.from_numpy(gray).to(self.device)
        t = t.float() / 255.0
        t = t[None, None, :, :]

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            pred = self.model.extract(t)

        # Keypoints + descriptors are consistent
        kpts = pred["keypoints"][0].detach().cpu().numpy().astype(np.float32)       # [N,2] (x,y)
        desc = pred["descriptors"][0].detach().cpu().numpy().astype(np.float32)     # [N,D]

        # Scores name differs across versions (or may be missing)
        if "scores" in pred:
            scr_t = pred["scores"][0]
        elif "keypoint_scores" in pred:
            scr_t = pred["keypoint_scores"][0]
        else:
            # If the extractor doesn't provide scores, just fill with 1s
            scr_t = torch.ones((kpts.shape[0],), device="cpu")

        scr = scr_t.detach().cpu().numpy().astype(np.float32)  # [N]

        return {"keypoints": kpts, "descriptors": desc, "scores": scr}



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--normalized-root",
        required=True,
        help="Path to outputs/normalized (contains poses_normalized.yaml + images/)",
    )
    ap.add_argument(
        "--dataset",
        default="ALL",
        help='Dataset/building name like "Campbell" or "Chem" or "ALL"',
    )
    ap.add_argument(
        "--out-root",
        default=None,
        help="Optional output root. Default: <normalized-root>/../features/",
    )
    ap.add_argument(
        "--max-side",
        type=int,
        default=1600,
        help="Resize so max(H,W) <= this. Use 0 to disable resizing.",
    )
    ap.add_argument(
        "--max-kpts",
        type=int,
        default=2048,
        help="Max keypoints per image.",
    )
    ap.add_argument(
        "--device",
        default=None,
        help='Force device: "cuda" or "cpu". Default auto.',
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing groups in the H5.",
    )
    args = ap.parse_args()

    max_side = None if args.max_side == 0 else args.max_side

    ext = SuperPointFeatureExtractor(
        normalized_root=Path(args.normalized_root),
        dataset=args.dataset,
        out_root=Path(args.out_root) if args.out_root else None,
        max_side=max_side,
        max_keypoints=args.max_kpts,
        device=args.device,
        overwrite=args.overwrite,
    )
    out_h5 = ext.run()
    print(f"[OK] wrote features: {out_h5}")


if __name__ == "__main__":
    main()
