from __future__ import annotations

import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
from PIL import Image, ExifTags
from tqdm import tqdm


@dataclass(frozen=True)
class ConvertResult:
    src: Path
    tiff: Path
    meta_yaml: Path


# -------------------------------------------------
# Worker init
# -------------------------------------------------
def _init_worker():
    # Register HEIC support inside each process
    try:
        import pillow_heif  # type: ignore
        pillow_heif.register_heif_opener()
    except Exception:
        pass


# -------------------------------------------------
# Image conversion + metadata (worker-safe)
# -------------------------------------------------
def _image_to_tiff(src: Path, dst: Path) -> Image.Image:
    try:
        img = Image.open(src)
        img_rgb = img.convert("RGB")
        img_rgb.save(dst, format="TIFF", compression="tiff_deflate")
        return img
    except Exception:
        magick = shutil.which("magick") or shutil.which("convert")
        if not magick:
            raise RuntimeError("Could not decode image with Pillow, and ImageMagick is not installed.")
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

            meta["exif"] = exif_readable
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

    meta_path.write_text(
        yaml.dump(meta, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    return (str(src), str(tiff_path), str(meta_path))


# -------------------------------------------------
# Main class
# -------------------------------------------------
class ImagesToTiffWithExistingMetadata:
    def __init__(self, image_dir: str | Path, out_dir: str | Path, workers: int | None = None) -> None:
        self.image_dir = Path(image_dir)
        self.out_dir = Path(out_dir)
        self.workers = workers or max(1, (os.cpu_count() or 2) - 1)

        self.out_images = self.out_dir / "images_tiff"
        self.out_meta = self.out_dir / "meta"
        self.out_images.mkdir(parents=True, exist_ok=True)
        self.out_meta.mkdir(parents=True, exist_ok=True)

    def convert_all(self) -> List[ConvertResult]:
        images = sorted(p for p in self.image_dir.iterdir() if p.is_file())
        if not images:
            return []

        tasks = [(str(p), str(self.out_images), str(self.out_meta)) for p in images]
        results: List[ConvertResult] = []

        with ProcessPoolExecutor(
            max_workers=self.workers,
            initializer=_init_worker,
        ) as executor:
            futures = [executor.submit(_process_one, t) for t in tasks]

            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Converting images",
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
                {
                    "src": str(r.src),
                    "tiff": str(r.tiff),
                    "meta_yaml": str(r.meta_yaml),
                }
                for r in sorted(results, key=lambda r: r.src.name)
            ],
        }

        (self.out_dir / "index.yaml").write_text(
            yaml.safe_dump(index, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )


# -------------------------------------------------
# Run
# -------------------------------------------------
if __name__ == "__main__":
    import sys

    image_dir = sys.argv[1]
    out_dir = sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) >= 4 else None

    ImagesToTiffWithExistingMetadata(
        image_dir=image_dir,
        out_dir=out_dir,
        workers=workers,
    ).convert_all()
