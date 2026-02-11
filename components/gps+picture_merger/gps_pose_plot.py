from __future__ import annotations

import csv
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from pyproj import CRS, Transformer

_AZIMUTH_DEG_BACKWARDS = [
    246.8, 245.9, 242.3, 236.3, 228.9, 220.1, 220.2,
    100.4, 99.2, 102.0, 93.9, 102.3, 115.7, 126.4,
    137.1, 139.5, 140.6, 148.7, 150.9, 153.8, 149.9,
    165.0, 178.0, 187.9, 180.5, 179.7, 191.2, 194.7,
    198.6, 222.7, 223.8, 226.1, 223.2, 224.4, 259.8,
    269.3, -88.8, -86.7, -86.9, -77.6, -69.5, -64.2, -59.4,
    -52.8, -54.7, -39.5, -41.5, -34.9, -24.0, -11.9, -6.2,
    -3.1, -1.2, 6.8, 16.8, 22.3, 30.0, 38.6,
    42.5, 47.3, 50.9, 59.5, 58.7, 65.3, 66.9,
    74.1, 83.4, 97.5, 95.8,
]

@dataclass(frozen=True)
class PoseRow:
    name: int
    lat_deg: float
    lon_deg: float
    h_ellip_m: float
    tilt_deg: float
    image_path: str  # string for YAML friendliness


class GPSPosesToYAML:
    """
    Load GNSS samples CSV + match images, convert LLH -> local ENU (meters),
    and write a YAML file with poses and placeholder direction vectors.

    Reference point:
      - If (ref_lat_deg, ref_lon_deg, ref_h_m) are provided: use them
      - Else: default to first pose in the CSV after parsing/matching
    """

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

        # Cache transformer (LLH->ECEF) once
        self._crs_llh = CRS.from_epsg(4979)   # WGS84 3D: lon/lat/ellip height
        self._crs_ecef = CRS.from_epsg(4978)  # WGS84 ECEF XYZ
        self._to_ecef = Transformer.from_crs(self._crs_llh, self._crs_ecef, always_xy=True)

        self._azimuth_deg = list(reversed(_AZIMUTH_DEG_BACKWARDS)) # Reversed to fix

    # -----------------------------
    # Public
    # -----------------------------
    def run(self) -> None:
        rows = self._load_csv_rows()
        if not rows:
            raise RuntimeError("No valid rows parsed from CSV.")

        poses = self._attach_images(rows)
        if not poses:
            raise RuntimeError("No poses after image matching.")

        ref_lat, ref_lon, ref_h = self._resolve_reference(poses)
        ref_llh = {"lat": ref_lat, "lon": ref_lon, "h_ellip_m": ref_h}

        out_poses: List[Dict] = []
        for idx, p in enumerate(poses):
            e, n, u = self._llh_to_local_enu(p.lat_deg, p.lon_deg, p.h_ellip_m, ref_lat, ref_lon, ref_h)

            az = self._get_azimuth_for_index(idx)

            out_poses.append(
                {
                    **asdict(p),
                    "enu_m": {"E": e, "N": n, "U": u},
                    "azimuth_deg": az,
                    "dir_enu": self._dir_enu_placeholder(tilt_deg=p.tilt_deg), # TODO: place holder
                }
            )

        out = {"reference_llh": ref_llh, "poses": out_poses}
        self.out_yaml.write_text(yaml.safe_dump(out, sort_keys=False), encoding="utf-8")

    # -----------------------------
    # Reference handling
    # -----------------------------
    def _resolve_reference(self, poses: List[PoseRow]) -> Tuple[float, float, float]:
        if self._ref_lat_deg is not None and self._ref_lon_deg is not None and self._ref_h_m is not None:
            return self._ref_lat_deg, self._ref_lon_deg, self._ref_h_m

        # default: first pose
        p0 = poses[0]
        return p0.lat_deg, p0.lon_deg, p0.h_ellip_m

    # -----------------------------
    # CSV + image matching
    # -----------------------------
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

    # -----------------------------
    # Coordinate helpers
    # -----------------------------
    def _llh_to_local_enu(
        self,
        lat_deg: float,
        lon_deg: float,
        h_m: float,
        ref_lat_deg: float,
        ref_lon_deg: float,
        ref_h_m: float,
    ) -> Tuple[float, float, float]:
        # LLH -> ECEF
        x, y, z = self._to_ecef.transform(lon_deg, lat_deg, h_m)
        x0, y0, z0 = self._to_ecef.transform(ref_lon_deg, ref_lat_deg, ref_h_m)

        # ECEF -> ENU (local tangent frame at reference)
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
        """
        Placeholder direction vector in ENU.
        TODO: Replace with real yaw/pitch/roll.
        Current behavior: yaw=0 (facing North), pitch-down from horizontal=tilt_deg, roll=0.
        """
        tilt = math.radians(tilt_deg)
        return {"dE": 0.0, "dN": math.cos(tilt), "dU": -math.sin(tilt)}


if __name__ == "__main__":
    import sys

    # Usage:
    #   python gps_to_yaml.py samples.csv images_tiff_dir out/poses.yaml
    #
    GPSPosesToYAML(sys.argv[1], sys.argv[2], sys.argv[3]).run()
