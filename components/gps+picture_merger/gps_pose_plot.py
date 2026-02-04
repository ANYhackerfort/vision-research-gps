from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import yaml


@dataclass(frozen=True)
class PoseRow:
    name: int
    lat_deg: float
    lon_deg: float
    h_ellip_m: float
    tilt_deg: float
    image_path: Path


# -----------------------------
# WGS84 helpers (no deps)
# -----------------------------
_WGS84_A = 6378137.0
_WGS84_F = 1 / 298.257223563
_WGS84_E2 = _WGS84_F * (2 - _WGS84_F)


def _llh_to_ecef(lat_deg: float, lon_deg: float, h: float) -> Tuple[float, float, float]:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)

    N = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
    x = (N + h) * cos_lat * cos_lon
    y = (N + h) * cos_lat * sin_lon
    z = (N * (1.0 - _WGS84_E2) + h) * sin_lat
    return x, y, z


def _ecef_to_enu(
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

    # E,N,U basis
    e = -sin_lon0 * dx + cos_lon0 * dy
    n = -sin_lat0 * cos_lon0 * dx - sin_lat0 * sin_lon0 * dy + cos_lat0 * dz
    u = cos_lat0 * cos_lon0 * dx + cos_lat0 * sin_lon0 * dy + sin_lat0 * dz
    return e, n, u


class GPSPosePlotter:
    """
    Reads a CSV of GNSS samples and plots camera centers in a local ENU frame.
    Also matches each CSV row Name -> image in images_tiff_dir (by number in filename or by order).
    """

    def __init__(self, csv_path: str | Path, images_tiff_dir: str | Path, out_dir: str | Path) -> None:
        self.csv_path = Path(csv_path)
        self.images_tiff_dir = Path(images_tiff_dir)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        rows = self._load_rows()
        if not rows:
            raise RuntimeError("No valid rows parsed from CSV.")

        poses = self._attach_images(rows)

        # Reference is first pose
        lat0, lon0, h0 = poses[0].lat_deg, poses[0].lon_deg, poses[0].h_ellip_m
        x0, y0, z0 = _llh_to_ecef(lat0, lon0, h0)

        enu_list: List[Dict] = []
        for p in poses:
            x, y, z = _llh_to_ecef(p.lat_deg, p.lon_deg, p.h_ellip_m)
            e, n, u = _ecef_to_enu(x, y, z, x0, y0, z0, lat0, lon0)

            # View direction arrow:
            # - assume yaw = 0 (facing North)
            # - tilt is pitch-down from horizontal
            # - arrow in ENU: North component + Up component
            tilt = math.radians(p.tilt_deg)
            arrow_len = 1.0  # meters (scaled up later)
            de = 0.0
            dn = math.cos(tilt) * arrow_len
            du = -math.sin(tilt) * arrow_len

            enu_list.append(
                {
                    "name": p.name,
                    "image": str(p.image_path),
                    "lat_deg": p.lat_deg,
                    "lon_deg": p.lon_deg,
                    "h_ellip_m": p.h_ellip_m,
                    "tilt_deg": p.tilt_deg,
                    "E_m": e,
                    "N_m": n,
                    "U_m": u,
                    "dir_ENU": {"dE": de, "dN": dn, "dU": du},
                }
            )

        (self.out_dir / "poses_enu.yaml").write_text(
            yaml.safe_dump({"reference": {"lat": lat0, "lon": lon0, "h": h0}, "poses": enu_list}, sort_keys=False),
            encoding="utf-8",
        )

        self._plot_3d(enu_list)
        self._plot_topdown(enu_list)

    # -----------------------------
    # CSV + image matching
    # -----------------------------
    def _load_rows(self) -> List[Tuple[int, float, float, float, float]]:
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
        tiffs = sorted([p for p in self.images_tiff_dir.iterdir() if p.is_file()])

        # map filename-number -> path, e.g. IMG_2245.tiff -> 2245
        num_to_path: Dict[int, Path] = {}
        for p in tiffs:
            n = self._last_int_in_stem(p.stem)
            if n is not None and n not in num_to_path:
                num_to_path[n] = p

        poses: List[PoseRow] = []
        for idx, (name, lat, lon, h, tilt) in enumerate(rows):
            img = num_to_path.get(name)
            if img is None:
                # fallback: by sorted order (row 1 -> first image, row 2 -> second image, ...)
                if 0 <= idx < len(tiffs):
                    img = tiffs[idx]
                else:
                    raise RuntimeError(f"Could not match Name={name} to any TIFF (and order fallback ran out).")

            poses.append(PoseRow(name=name, lat_deg=lat, lon_deg=lon, h_ellip_m=h, tilt_deg=tilt, image_path=img))

        return poses

    def _last_int_in_stem(self, stem: str) -> Optional[int]:
        # extract last integer group from a filename stem
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
    # Plotting
    # -----------------------------
    def _plot_3d(self, poses: List[Dict]) -> None:
        E = [p["E_m"] for p in poses]
        N = [p["N_m"] for p in poses]
        U = [p["U_m"] for p in poses]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(E, N, U)

        # scale arrows to look nice relative to spread
        span = max(max(E) - min(E), max(N) - min(N), max(U) - min(U), 1.0)
        scale = 0.15 * span

        for p in poses:
            e, n, u = p["E_m"], p["N_m"], p["U_m"]
            dE = p["dir_ENU"]["dE"] * scale
            dN = p["dir_ENU"]["dN"] * scale
            dU = p["dir_ENU"]["dU"] * scale
            ax.quiver(e, n, u, dE, dN, dU, length=1.0, normalize=False)

        ax.set_xlabel("E (m)")
        ax.set_ylabel("N (m)")
        ax.set_zlabel("U (m)")
        ax.set_title("Camera centers in local ENU (tilt arrows assume facing North)")

        fig.tight_layout()
        fig.savefig(self.out_dir / "gps_3d.png", dpi=200)
        plt.close(fig)

    def _plot_topdown(self, poses: List[Dict]) -> None:
        E = [p["E_m"] for p in poses]
        N = [p["N_m"] for p in poses]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(E, N)

        span = max(max(E) - min(E), max(N) - min(N), 1.0)
        scale = 0.15 * span

        for p in poses:
            e, n = p["E_m"], p["N_m"]
            dE = p["dir_ENU"]["dE"] * scale
            dN = p["dir_ENU"]["dN"] * scale
            ax.arrow(e, n, dE, dN, head_width=0.03 * span, length_includes_head=True)

        ax.set_xlabel("E (m)")
        ax.set_ylabel("N (m)")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title("Top-down EN plot (tilt arrows projected, assume facing North)")

        fig.tight_layout()
        fig.savefig(self.out_dir / "gps_topdown.png", dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    import sys

    csv_path = sys.argv[1]
    images_tiff_dir = sys.argv[2]
    out_dir = sys.argv[3]

    GPSPosePlotter(csv_path, images_tiff_dir, out_dir).run()
