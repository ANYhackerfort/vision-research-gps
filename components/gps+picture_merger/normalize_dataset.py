from __future__ import annotations

import csv
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


@dataclass(frozen=True)
class PoseRow:
    dataset: str
    name: int
    image_src: Path
    image_canon: str
    E: float
    N: float
    U: float
    lat_deg: Optional[float] = None
    lon_deg: Optional[float] = None
    h_ellip_m: Optional[float] = None
    tilt_deg: Optional[float] = None


class DatasetNormalizer:
    """
    Normalizes:
      - images into outputs/normalized/images/
      - poses into outputs/normalized/poses_normalized.yaml
      - ENU table into outputs/normalized/cameras.csv
      - mapping into outputs/normalized/image_map.csv

    Supports either:
      - a single poses.yaml path
      - or a root folder that contains multiple */poses.yaml
    """

    def __init__(
        self,
        root: Path,
        out_dir: Optional[Path] = None,
        link_mode: str = "copy",  # "copy" | "symlink"
        image_exts: Tuple[str, ...] = (".tif", ".tiff"),
    ) -> None:
        self.root = Path(root)
        self.out_dir = Path(out_dir) if out_dir else self.root / "outputs" / "normalized"
        self.images_dir = self.out_dir / "images"
        self.link_mode = link_mode
        self.image_exts = tuple(e.lower() for e in image_exts)

    # -----------------------------
    # Public API
    # -----------------------------
    def run(self) -> List[PoseRow]:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)

        pose_files = self._discover_pose_files(self.root)
        if not pose_files:
            raise RuntimeError(f"No poses.yaml found under: {self.root}")

        rows: List[PoseRow] = []
        normalized_docs: List[Dict] = []

        for ypath in pose_files:
            dataset_name = ypath.parent.name
            doc = yaml.safe_load(ypath.read_text()) or {}
            poses = doc.get("poses", []) or []

            norm_doc = {
                "source_poses_yaml": str(ypath),
                "reference_llh": doc.get("reference_llh", None),
                "poses": [],
            }

            for p in poses:
                row = self._normalize_one_pose(dataset_name, p)
                rows.append(row)

                # write normalized pose entry
                norm_doc["poses"].append(
                    {
                        "dataset": row.dataset,
                        "name": row.name,
                        "image_name": row.image_canon,
                        "image_path": str(self.images_dir / row.image_canon),
                        "enu_m": {"E": row.E, "N": row.N, "U": row.U},
                        "lat_deg": row.lat_deg,
                        "lon_deg": row.lon_deg,
                        "h_ellip_m": row.h_ellip_m,
                        "tilt_deg": row.tilt_deg,
                    }
                )

            normalized_docs.append(norm_doc)

        # Write outputs
        self._write_cameras_csv(rows, self.out_dir / "cameras.csv")
        self._write_image_map_csv(rows, self.out_dir / "image_map.csv")

        # If multiple pose yamls, write a single multi-scene yaml
        out_yaml = self.out_dir / "poses_normalized.yaml"
        out_yaml.write_text(yaml.safe_dump({"scenes": normalized_docs}, sort_keys=False))

        return rows

    # -----------------------------
    # Discovery
    # -----------------------------
    def _discover_pose_files(self, root: Path) -> List[Path]:
        root = Path(root)
        if root.is_file() and root.name == "poses.yaml":
            return [root]
        # otherwise search
        return sorted(root.rglob("poses.yaml"))

    # -----------------------------
    # Pose normalization
    # -----------------------------
    def _normalize_one_pose(self, dataset: str, p: Dict) -> PoseRow:
        name = int(p["name"])
        img_src = Path(p["image_path"]).expanduser()

        if not img_src.exists():
            raise RuntimeError(f"Missing image file: {img_src}")

        # Canonical image name:
        # Prefer original stem, but prefix dataset + pose id to avoid collisions across scenes.
        # Example: Campbell_0001_IMG_5839.tiff
        orig_name = img_src.name
        canon = f"{dataset}_{name:04d}_{orig_name}"

        # Enforce extension sanity (optional)
        ext = img_src.suffix.lower()
        if ext not in self.image_exts:
            # allow it anyway, but you can tighten later
            pass

        # Link/copy into normalized images folder
        dst = self.images_dir / canon
        self._materialize_image(img_src, dst)

        enu = (p.get("enu_m") or {})
        E = float(enu["E"])
        N = float(enu["N"])
        U = float(enu["U"])

        return PoseRow(
            dataset=dataset,
            name=name,
            image_src=img_src,
            image_canon=canon,
            E=E,
            N=N,
            U=U,
            lat_deg=_maybe_float(p.get("lat_deg")),
            lon_deg=_maybe_float(p.get("lon_deg")),
            h_ellip_m=_maybe_float(p.get("h_ellip_m")),
            tilt_deg=_maybe_float(p.get("tilt_deg")),
        )

    def _materialize_image(self, src: Path, dst: Path) -> None:
        if dst.exists():
            return  # already done

        if self.link_mode == "symlink":
            # Relative symlink is nicer if possible
            try:
                rel = os.path.relpath(src, start=dst.parent)
                dst.symlink_to(rel)
            except Exception:
                dst.symlink_to(src)
        elif self.link_mode == "copy":
            shutil.copy2(src, dst)
        else:
            raise ValueError(f"Unknown link_mode={self.link_mode!r} (use 'copy' or 'symlink')")

    # -----------------------------
    # CSV writers
    # -----------------------------
    def _write_cameras_csv(self, rows: List[PoseRow], out_csv: Path) -> None:
        fields = [
            "dataset",
            "name",
            "image_name",
            "image_path",
            "E",
            "N",
            "U",
            "lat_deg",
            "lon_deg",
            "h_ellip_m",
            "tilt_deg",
        ]
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow(
                    {
                        "dataset": r.dataset,
                        "name": r.name,
                        "image_name": r.image_canon,
                        "image_path": str(self.images_dir / r.image_canon),
                        "E": r.E,
                        "N": r.N,
                        "U": r.U,
                        "lat_deg": r.lat_deg if r.lat_deg is not None else "",
                        "lon_deg": r.lon_deg if r.lon_deg is not None else "",
                        "h_ellip_m": r.h_ellip_m if r.h_ellip_m is not None else "",
                        "tilt_deg": r.tilt_deg if r.tilt_deg is not None else "",
                    }
                )

    def _write_image_map_csv(self, rows: List[PoseRow], out_csv: Path) -> None:
        fields = ["dataset", "name", "image_name", "normalized_path", "source_path"]
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow(
                    {
                        "dataset": r.dataset,
                        "name": r.name,
                        "image_name": r.image_canon,
                        "normalized_path": str(self.images_dir / r.image_canon),
                        "source_path": str(r.image_src),
                    }
                )


def _maybe_float(x) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


if __name__ == "__main__":
    # Example:
    #   python3 normalize_dataset.py /path/to/pose_viz_out/Campbell
    # or:
    #   python3 normalize_dataset.py /path/to/pose_viz_out
    import sys

    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    norm = DatasetNormalizer(root=root, link_mode="copy")  # change to "symlink" if you want
    rows = norm.run()
    print(f"[OK] normalized {len(rows)} poses into: {norm.out_dir}")
