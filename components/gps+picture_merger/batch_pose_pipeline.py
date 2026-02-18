#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from PIL import Image, ExifTags
from tqdm import tqdm
from pyproj import CRS, Transformer

# ============================================================
# GLOBAL REFERENCE (UCSB-ish campus center)
# Swap these if you have the exact plaque coordinate.
# ============================================================
UCSB_REF_LAT_DEG = 34.413963   # :contentReference[oaicite:1]{index=1}
UCSB_REF_LON_DEG = -119.848946 # :contentReference[oaicite:2]{index=2}

# Height: you can keep dataset-relative height by using the first pose's height.
# If you want a fixed absolute height instead, set this to a number (meters).
UCSB_REF_H_MODE = "use_first_pose"  # or "fixed"
UCSB_REF_H_M = 0.0                 # used only if UCSB_REF_H_MODE == "fixed"

# ============================================================
# Your azimuth list (TODO: Replace with actual Azimuth file)
# ============================================================
_AZIMUTH_DEG_ = [
    95.8, 97.5, 83.4, 74.1, 66.9, 65.3, 58.7, 59.5, 50.9, 47.3,
    42.5, 38.6, 30.0, 22.3, 16.8, 6.8, -1.2, -3.1, -6.2, -11.9,
    -24.0, -34.9, -41.5, -39.5, -54.7, -52.8, -59.4, -64.2,
    -69.5, -77.6, -86.9, -86.7, -88.8, 269.3, 259.8, 224.4,
    223.2, 226.1, 223.8, 222.7, 198.6, 194.7, 191.2, 179.7,
    180.5, 187.9, 178.0, 165.0, 149.9, 153.8, 150.9, 148.7,
    140.6, 139.5, 137.1, 126.4, 115.7, 102.3, 93.9, 102.0,
    99.2, 100.4, 220.2, 220.1, 228.9, 236.3, 242.3, 245.9, 246.8
]


# ============================================================
# (A) Convert images -> TIFF + metadata
# ============================================================
@dataclass(frozen=True)
class ConvertResult:
    src: Path
    tiff: Path
    meta_yaml: Path


def _init_worker():
    try:
        import pillow_heif  # type: ignore
        pillow_heif.register_heif_opener()
    except Exception:
        pass


def _image_to_tiff(src: Path, dst: Path) -> Image.Image:
    try:
        img = Image.open(src)
        img_rgb = img.convert("RGB")
        img_rgb.save(dst, format="TIFF", compression="tiff_deflate")
        return img
    except Exception:
        magick = shutil.which("magick") or shutil.which("convert")
        if not magick:
            raise RuntimeError(
                f"Could not decode {src.name} with Pillow, and ImageMagick is not installed."
            )
        subprocess.run([magick, str(src), str(dst)], check=True)
        return Image.open(dst)


def _extract_metadata(img: Image.Image, src_path: Path, tiff_path: Path) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "file": {
            "src_filename": src_path.name,
            "src_path": str(src_path.resolve()),
            "tiff_filename": tiff_path.name,
            "tiff_path": str(tiff_path.resolve()),
        },
        "exif": {},
    }

    try:
        raw_exif = img.getexif()
        if raw_exif:
            exif_readable: Dict[str, Any] = {}
            for tag_id, value in raw_exif.items():
                tag = ExifTags.TAGS.get(tag_id, str(tag_id))
                exif_readable[tag] = value

            gps = exif_readable.get("GPSInfo")
            if isinstance(gps, dict):
                gps_decoded: Dict[str, Any] = {}
                for k, v in gps.items():
                    gps_tag = ExifTags.GPSTAGS.get(k, str(k))
                    gps_decoded[gps_tag] = v
                exif_readable["GPSInfo"] = gps_decoded

            meta["exif"] = _yaml_safe(exif_readable)
    except Exception:
        pass

    try:
        meta["image"] = {"mode": img.mode, "size": list(img.size)}
    except Exception:
        pass

    return meta


def _process_one(args: Tuple[str, str, str]) -> Tuple[str, str, str]:
    src_s, out_images_s, out_meta_s = args
    src = Path(src_s)
    out_images = Path(out_images_s)
    out_meta = Path(out_meta_s)

    tiff_path = out_images / f"{src.stem}.tiff"
    meta_path = out_meta / f"{src.stem}.yaml"

    img = _image_to_tiff(src, tiff_path)
    meta = _extract_metadata(img, src, tiff_path)

    meta_safe = _yaml_safe(meta)
    meta_path.write_text(
        yaml.safe_dump(meta_safe, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    return (str(src), str(tiff_path), str(meta_path))

def _yaml_safe(obj):
    """
    Convert PIL/EXIF/numpy-ish objects into YAML-safe primitives.
    Handles tuples, bytes, dicts, lists, and PIL IFDRational / Fraction-like.
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # bytes -> text
    if isinstance(obj, (bytes, bytearray)):
        try:
            return obj.decode("utf-8", errors="replace")
        except Exception:
            return str(obj)

    # tuple/set -> list
    if isinstance(obj, (tuple, set)):
        return [_yaml_safe(x) for x in obj]

    # list -> list
    if isinstance(obj, list):
        return [_yaml_safe(x) for x in obj]

    # dict -> dict (stringify keys if needed)
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            kk = k if isinstance(k, (str, int, float, bool)) else str(k)
            out[kk] = _yaml_safe(v)
        return out

    # PIL rational / Fraction-like: try float()
    try:
        # Some EXIF types support numerator/denominator or float conversion
        return float(obj)
    except Exception:
        pass

    # fallback: stringify
    return str(obj)


class ImagesToTiffWithExistingMetadata:
    SUPPORTED_EXTS = {".heic", ".heif", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}

    def __init__(self, image_dir: str | Path, out_dir: str | Path, workers: int | None = None) -> None:
        self.image_dir = Path(image_dir)
        self.out_dir = Path(out_dir)
        self.workers = workers or max(1, (os.cpu_count() or 2) - 1)

        self.out_images = self.out_dir / "images_tiff"
        self.out_meta = self.out_dir / "meta"
        self.out_images.mkdir(parents=True, exist_ok=True)
        self.out_meta.mkdir(parents=True, exist_ok=True)

    def convert_all(self) -> List[ConvertResult]:
        images = sorted(
            p for p in self.image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in self.SUPPORTED_EXTS
        )
        if not images:
            return []

        tasks = [(str(p), str(self.out_images), str(self.out_meta)) for p in images]
        results: List[ConvertResult] = []

        with ProcessPoolExecutor(max_workers=self.workers, initializer=_init_worker) as executor:
            futures = [executor.submit(_process_one, t) for t in tasks]
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Converting {self.image_dir.name}",
                unit="img",
            ):
                src_s, tiff_s, meta_s = fut.result()
                results.append(ConvertResult(Path(src_s), Path(tiff_s), Path(meta_s)))

        self._write_index(results)
        return results

    def _write_index(self, results: List[ConvertResult]) -> None:
        index = {
            "generated_at": datetime.now().isoformat(),
            "workers": self.workers,
            "items": [
                {"src": str(r.src), "tiff": str(r.tiff), "meta_yaml": str(r.meta_yaml)}
                for r in sorted(results, key=lambda r: r.src.name)
            ],
        }
        (self.out_dir / "index.yaml").write_text(
            yaml.safe_dump(index, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )


# ============================================================
# (B) GPSPosesToYAML (CSV-only)
# ============================================================
@dataclass(frozen=True)
class PoseRow:
    name: int
    lat_deg: float
    lon_deg: float
    h_ellip_m: float
    tilt_deg: float
    image_path: str


class GPSPosesToYAML:
    def __init__(
        self,
        csv_path: str | Path,
        images_tiff_dir: str | Path,
        out_yaml: str | Path,
        *,
        ref_lat_deg: float | None = None,
        ref_lon_deg: float | None = None,
        ref_h_m: float | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.images_dir = Path(images_tiff_dir)
        self.out_yaml = Path(out_yaml)
        self.out_yaml.parent.mkdir(parents=True, exist_ok=True)

        self._ref_lat_deg = ref_lat_deg
        self._ref_lon_deg = ref_lon_deg
        self._ref_h_m = ref_h_m

        self._crs_llh = CRS.from_epsg(4979)
        self._crs_ecef = CRS.from_epsg(4978)
        self._to_ecef = Transformer.from_crs(self._crs_llh, self._crs_ecef, always_xy=True)

        self._azimuth_deg = list(_AZIMUTH_DEG_)

    def run(self) -> None:
        rows = self._load_csv_rows()
        if not rows:
            raise RuntimeError(f"No valid rows parsed from CSV: {self.csv_path}")

        poses = self._attach_images(rows)
        if not poses:
            raise RuntimeError("No poses after image matching.")

        ref_lat, ref_lon, ref_h = self._resolve_reference(poses)
        ref_llh = {"lat": ref_lat, "lon": ref_lon, "h_ellip_m": ref_h}

        out_poses: List[Dict] = []
        for idx, p in enumerate(poses):
            e, n, u = self._llh_to_local_enu(p.lat_deg, p.lon_deg, p.h_ellip_m, ref_lat, ref_lon, ref_h)
            # az = self._get_azimuth_for_index(idx)

            out_poses.append(
                {
                    **asdict(p),
                    "enu_m": {"E": e, "N": n, "U": u},
                    # "azimuth_deg": az,
                    "dir_enu": self._dir_enu_placeholder(tilt_deg=p.tilt_deg),
                }
            )

        out = {"reference_llh": ref_llh, "poses": out_poses}
        self.out_yaml.write_text(yaml.safe_dump(out, sort_keys=False), encoding="utf-8")

    def _resolve_reference(self, poses: List[PoseRow]) -> Tuple[float, float, float]:
        # If user explicitly passed a reference, honor it
        if self._ref_lat_deg is not None and self._ref_lon_deg is not None and self._ref_h_m is not None:
            return self._ref_lat_deg, self._ref_lon_deg, self._ref_h_m

        # Otherwise use global UCSB reference for lat/lon
        ref_lat = UCSB_REF_LAT_DEG
        ref_lon = UCSB_REF_LON_DEG

        # Height choice:
        # - "use_first_pose": keeps U roughly relative per dataset (usually what you want)
        # - "fixed": absolute U relative to UCSB_REF_H_M (if you really want that)
        if UCSB_REF_H_MODE == "fixed":
            ref_h = float(UCSB_REF_H_M)
        else:
            ref_h = float(poses[0].h_ellip_m)

        return ref_lat, ref_lon, ref_h

    def _load_csv_rows(self) -> List[Tuple[int, float, float, float, float]]:
        out: List[Tuple[int, float, float, float, float]] = []
        with self.csv_path.open("r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                name_s = (row.get("Name") or "").strip()
                lat_s = (row.get("Latitude") or "").strip()
                lon_s = (row.get("Longitude") or "").strip()
                h_s = (row.get("Ellipsoidal height") or "").strip()
                tilt_s = (row.get("Tilt angle") or "").strip()

                if not name_s or not lat_s or not lon_s or not h_s:
                    continue

                try:
                    name = int(float(name_s))
                    lat = float(lat_s)
                    lon = float(lon_s)
                    h = float(h_s)
                    tilt = float(tilt_s) if tilt_s else 0.0
                except Exception:
                    continue

                out.append((name, lat, lon, h, tilt))

        out.sort(key=lambda x: x[0])
        return out

    def _attach_images(self, rows: List[Tuple[int, float, float, float, float]]) -> List[PoseRow]:
        tiffs = sorted([p for p in self.images_dir.iterdir() if p.is_file()])

        num_to_path: Dict[int, Path] = {}
        for p in tiffs:
            n = self._last_int_in_stem(p.stem)
            if n is not None and n not in num_to_path:
                num_to_path[n] = p

        poses: List[PoseRow] = []
        for idx, (name, lat, lon, h, tilt) in enumerate(rows):
            img = num_to_path.get(name) or (tiffs[idx] if idx < len(tiffs) else None)
            if img is None:
                raise RuntimeError(f"Could not match Name={name} to any image in {self.images_dir}")

            poses.append(
                PoseRow(
                    name=name,
                    lat_deg=lat,
                    lon_deg=lon,
                    h_ellip_m=h,
                    tilt_deg=tilt,
                    image_path=str(img),
                )
            )
        return poses

    def _last_int_in_stem(self, stem: str) -> Optional[int]:
        cur = ""
        last = None
        for ch in stem:
            if ch.isdigit():
                cur += ch
            else:
                if cur:
                    last = int(cur)
                    cur = ""
        if cur:
            last = int(cur)
        return last

    def _llh_to_local_enu(
        self,
        lat_deg: float,
        lon_deg: float,
        h_m: float,
        ref_lat_deg: float,
        ref_lon_deg: float,
        ref_h_m: float,
    ) -> Tuple[float, float, float]:
        x, y, z = self._to_ecef.transform(lon_deg, lat_deg, h_m)
        x0, y0, z0 = self._to_ecef.transform(ref_lon_deg, ref_lat_deg, ref_h_m)
        return self._ecef_to_enu(x, y, z, x0, y0, z0, ref_lat_deg, ref_lon_deg)

    def _ecef_to_enu(
        self,
        x: float,
        y: float,
        z: float,
        x0: float,
        y0: float,
        z0: float,
        lat0_deg: float,
        lon0_deg: float,
    ) -> Tuple[float, float, float]:
        lat0 = math.radians(lat0_deg)
        lon0 = math.radians(lon0_deg)

        dx = x - x0
        dy = y - y0
        dz = z - z0

        sin_lat0 = math.sin(lat0)
        cos_lat0 = math.cos(lat0)
        sin_lon0 = math.sin(lon0)
        cos_lon0 = math.cos(lon0)

        e = -sin_lon0 * dx + cos_lon0 * dy
        n = -sin_lat0 * cos_lon0 * dx - sin_lat0 * sin_lon0 * dy + cos_lat0 * dz
        u = cos_lat0 * cos_lon0 * dx + cos_lat0 * sin_lon0 * dy + sin_lat0 * dz
        return e, n, u

    def _get_azimuth_for_index(self, idx: int) -> float:
        if idx >= len(self._azimuth_deg):
            raise IndexError("Azimuth list shorter than number of poses.")
        return self._azimuth_deg[idx]

    def _dir_enu_placeholder(self, *, tilt_deg: float) -> Dict[str, float]:
        tilt = math.radians(tilt_deg)
        return {"dE": 0.0, "dN": math.cos(tilt), "dU": -math.sin(tilt)}


# ============================================================
# (C) Batch orchestration (CSV-only)
# ============================================================
def _find_first_csv(folder: Path) -> Optional[Path]:
    csvs = sorted(folder.glob("*.csv"))
    return csvs[0] if csvs else None


def process_one_dataset(dataset_dir: Path, out_root: Path, workers: Optional[int]) -> None:
    out_dir = out_root / dataset_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    converter = ImagesToTiffWithExistingMetadata(dataset_dir, out_dir, workers=workers)
    results = converter.convert_all()
    if not results:
        print(f"[skip] {dataset_dir}: no images found")
        return

    csv_path = _find_first_csv(dataset_dir)
    if csv_path is None:
        print(f"[skip] {dataset_dir}: no .csv found")
        return

    out_yaml = out_dir / "poses.yaml"
    GPSPosesToYAML(
        csv_path=csv_path,
        images_tiff_dir=out_dir / "images_tiff",
        out_yaml=out_yaml,
    ).run()

    print(f"[ok] {dataset_dir.name} -> {out_yaml}")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("pictures_root", type=str, help="Root dir containing dataset folders (e.g. pictures/)")
    ap.add_argument("out_root", type=str, help="Output root directory (e.g. pose_viz_out/)")
    ap.add_argument("--workers", type=int, default=None, help="Process workers (default: cpu_count-1)")
    args = ap.parse_args()

    pics = Path(args.pictures_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    subdirs = sorted([p for p in pics.iterdir() if p.is_dir()])
    if not subdirs:
        raise SystemExit(f"No subfolders found under: {pics}")

    for d in subdirs:
        process_one_dataset(d, out_root, args.workers)


if __name__ == "__main__":
    main()
