from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import plotly.graph_objects as go
import yaml
from plyfile import PlyData  # pip install plyfile

# ============================================================
# ENU reference (your "UCSB center")
# ============================================================
UCSB_REF_LAT_DEG = 34.413963
UCSB_REF_LON_DEG = -119.848946
EARTH_R_M = 6378137.0


# ------------------------------------------------------------
# Pose container
# ------------------------------------------------------------
@dataclass(frozen=True)
class PoseItem:
    dataset: str
    name: int
    E: float
    N: float
    U: float
    tilt_deg: float
    azimuth_deg: float


# ------------------------------------------------------------
# Multi YAML loader
# ------------------------------------------------------------
def load_all_pose_yamls(root: Path) -> List[PoseItem]:
    poses: List[PoseItem] = []
    yaml_files = sorted(root.rglob("poses.yaml"))
    if not yaml_files:
        raise RuntimeError(f"No poses.yaml files found under {root}")

    for yfile in yaml_files:
        dataset_name = yfile.parent.name
        data = yaml.safe_load(yfile.read_text()) or {}
        for p in data.get("poses", []):
            enu = p.get("enu_m") or {}
            poses.append(
                PoseItem(
                    dataset=dataset_name,
                    name=int(p["name"]),
                    E=float(enu.get("E", 0.0)),
                    N=float(enu.get("N", 0.0)),
                    U=float(enu.get("U", 0.0)),
                    tilt_deg=float(p.get("tilt_deg", 0.0)),
                    azimuth_deg=float(p.get("azimuth_deg", 0.0)),
                )
            )
    return poses


# ------------------------------------------------------------
# Vector helpers (in ENU)
# ------------------------------------------------------------
def unit(x: float, y: float, z: float) -> Tuple[float, float, float]:
    n = math.sqrt(x * x + y * y + z * z)
    if n <= 0:
        return 0.0, 0.0, 0.0
    return x / n, y / n, z / n


def azimuth_dir(azimuth_deg: float) -> Tuple[float, float, float]:
    az = math.radians(azimuth_deg)
    return unit(math.sin(az), math.cos(az), 0.0)


def tilt_dir(azimuth_deg: float, tilt_deg: float) -> Tuple[float, float, float]:
    az = math.radians(azimuth_deg)
    tilt = math.radians(tilt_deg)
    horiz = math.cos(tilt)
    dE = math.sin(az) * horiz
    dN = math.cos(az) * horiz
    dU = math.sin(tilt)
    return unit(dE, dN, dU)


# ------------------------------------------------------------
# ENU -> Lat/Lon (local approximation)
# ------------------------------------------------------------
def enu_to_latlon(E_m: float, N_m: float, ref_lat_deg: float, ref_lon_deg: float) -> Tuple[float, float]:
    lat0 = math.radians(ref_lat_deg)
    dlat = N_m / EARTH_R_M
    dlon = E_m / (EARTH_R_M * math.cos(lat0))
    lat = ref_lat_deg + math.degrees(dlat)
    lon = ref_lon_deg + math.degrees(dlon)
    return lat, lon


# ------------------------------------------------------------
# PLY loader (binary or ascii) + optional RGB
# ------------------------------------------------------------
def _first_present(names: Tuple[str, ...], dtype_names: Tuple[str, ...]) -> Optional[str]:
    for n in names:
        if n in dtype_names:
            return n
    return None


def load_ply_vertices(ply_path: Path) -> Dict[str, Any]:
    ply = PlyData.read(str(ply_path))
    if "vertex" not in ply:
        raise RuntimeError(f"No vertex element in PLY: {ply_path}")

    v = ply["vertex"]
    names = v.data.dtype.names or ()
    for key in ("x", "y", "z"):
        if key not in names:
            raise RuntimeError(f"PLY missing vertex '{key}' field: {ply_path}")

    r_name = _first_present(("red", "r"), names)
    g_name = _first_present(("green", "g"), names)
    b_name = _first_present(("blue", "b"), names)
    has_rgb = (r_name is not None and g_name is not None and b_name is not None)

    xs = v["x"]
    ys = v["y"]
    zs = v["z"]

    xyz: List[Tuple[float, float, float]] = []
    rgb: Optional[List[Tuple[int, int, int]]] = [] if has_rgb else None

    if has_rgb:
        rs = v[r_name]  # type: ignore[index]
        gs = v[g_name]  # type: ignore[index]
        bs = v[b_name]  # type: ignore[index]

    for i in range(len(xs)):
        xyz.append((float(xs[i]), float(ys[i]), float(zs[i])))
        if has_rgb and rgb is not None:
            rgb.append((int(rs[i]), int(gs[i]), int(bs[i])))

    return {"xyz": xyz, "rgb": rgb}


def add_ply_to_map(
    fig: go.Figure,
    ply_path: Path,
    *,
    name: str = "PLY",
    sample_n: int = 50000,
    size: int = 7,          # BIGGER
    opacity: float = 0.95,
) -> Tuple[List[float], List[float]]:
    data = load_ply_vertices(ply_path)
    xyz: List[Tuple[float, float, float]] = data["xyz"]
    rgb: Optional[List[Tuple[int, int, int]]] = data["rgb"]

    n = len(xyz)
    if n == 0:
        print("[warn] PLY has 0 vertices")
        return [], []

    if sample_n > 0 and n > sample_n:
        idxs = random.sample(range(n), sample_n)
        xyz = [xyz[i] for i in idxs]
        if rgb is not None:
            rgb = [rgb[i] for i in idxs]

    lats: List[float] = []
    lons: List[float] = []
    hovers: List[str] = []

    for (E, N, U) in xyz:
        lat, lon = enu_to_latlon(E, N, UCSB_REF_LAT_DEG, UCSB_REF_LON_DEG)
        lats.append(lat)
        lons.append(lon)
        hovers.append(f"ENU(m): E={E:.2f}, N={N:.2f}, U={U:.2f}")

    marker_kwargs: Dict[str, Any] = dict(size=size, opacity=opacity)
    if rgb is not None:
        marker_kwargs["color"] = [f"rgb({r},{g},{b})" for (r, g, b) in rgb]
    else:
        marker_kwargs["color"] = "white"

    fig.add_trace(
        go.Scattermapbox(
            lat=lats,
            lon=lons,
            mode="markers",
            marker=marker_kwargs,
            name=name,
            hovertext=hovers,
            hoverinfo="text",
        )
    )

    return lats, lons


# ------------------------------------------------------------
# Manual "fit bounds" for older Plotly versions
# ------------------------------------------------------------
def bounds_from_points(
    lat_lists: List[List[float]],
    lon_lists: List[List[float]],
    *,
    pad_frac: float = 0.08,
) -> Optional[Dict[str, float]]:
    lats = [x for lst in lat_lists for x in lst]
    lons = [x for lst in lon_lists for x in lst]
    if not lats or not lons:
        return None

    south = min(lats)
    north = max(lats)
    west = min(lons)
    east = max(lons)

    # pad
    lat_span = max(north - south, 1e-9)
    lon_span = max(east - west, 1e-9)

    south -= lat_span * pad_frac
    north += lat_span * pad_frac
    west -= lon_span * pad_frac
    east += lon_span * pad_frac

    return {"south": south, "north": north, "west": west, "east": east}


# ------------------------------------------------------------
# Map viewer
# ------------------------------------------------------------
def build_map_view(root_dir: str | Path, out_html: str | Path, ply_path: Optional[str | Path] = None):
    root = Path(root_dir)
    out_html = Path(out_html)

    poses = load_all_pose_yamls(root)
    datasets = sorted(set(p.dataset for p in poses))

    colors = ["red", "blue", "green", "orange", "purple", "cyan", "magenta", "brown"]

    fig = go.Figure()

    # We collect lat/lon lists from everything we plot, so we can set bounds.
    all_lat_lists: List[List[float]] = []
    all_lon_lists: List[List[float]] = []

    # UCSB reference marker
    fig.add_trace(
        go.Scattermapbox(
            lat=[UCSB_REF_LAT_DEG],
            lon=[UCSB_REF_LON_DEG],
            mode="markers+text",
            text=["UCSB_REF (ENU origin)"],
            textposition="top center",
            marker=dict(size=18, color="white", opacity=0.95),
            name="UCSB reference",
            hovertext=[f"ENU origin<br>lat={UCSB_REF_LAT_DEG:.6f}, lon={UCSB_REF_LON_DEG:.6f}"],
            hoverinfo="text",
        )
    )
    all_lat_lists.append([UCSB_REF_LAT_DEG])
    all_lon_lists.append([UCSB_REF_LON_DEG])

    # PLY layer (preserve colors, make bigger)
    if ply_path is not None:
        ply_path = Path(ply_path)
        if ply_path.exists():
            ply_lats, ply_lons = add_ply_to_map(
                fig,
                ply_path,
                name=ply_path.name,
                sample_n=50000,
                size=7,
                opacity=0.95,
            )
            all_lat_lists.append(ply_lats)
            all_lon_lists.append(ply_lons)
        else:
            print(f"[warn] PLY path not found, skipping: {ply_path}")

    AZ_LEN_M = 10.0
    TILT_GROUND_LEN_M = 15.0

    for idx, dataset in enumerate(datasets):
        subset = [p for p in poses if p.dataset == dataset]
        color = colors[idx % len(colors)]

        lats, lons, hovers, labels = [], [], [], []
        for p in subset:
            lat, lon = enu_to_latlon(p.E, p.N, UCSB_REF_LAT_DEG, UCSB_REF_LON_DEG)
            lats.append(lat)
            lons.append(lon)
            labels.append(f"{dataset}:{p.name}")
            hovers.append(
                f"{dataset}:{p.name}"
                f"<br>ENU(m): E={p.E:.2f}, N={p.N:.2f}, U={p.U:.2f}"
                f"<br>az={p.azimuth_deg:.1f}°, tilt={p.tilt_deg:.1f}°"
                f"<br>lat={lat:.6f}, lon={lon:.6f}"
            )

        fig.add_trace(
            go.Scattermapbox(
                lat=lats,
                lon=lons,
                mode="markers+text",
                text=labels,
                textposition="top right",
                marker=dict(size=16, color=color, opacity=0.95),  # BIGGER
                name=f"{dataset} poses",
                hovertext=hovers,
                hoverinfo="text",
            )
        )
        all_lat_lists.append(lats)
        all_lon_lists.append(lons)

        # (lines don't affect bounds much, but that's fine)

        az_lat, az_lon = [], []
        for p in subset:
            lat0, lon0 = enu_to_latlon(p.E, p.N, UCSB_REF_LAT_DEG, UCSB_REF_LON_DEG)
            dE, dN, _ = azimuth_dir(p.azimuth_deg)
            lat1, lon1 = enu_to_latlon(p.E + dE * AZ_LEN_M, p.N + dN * AZ_LEN_M, UCSB_REF_LAT_DEG, UCSB_REF_LON_DEG)
            az_lat += [lat0, lat1, None]
            az_lon += [lon0, lon1, None]

        fig.add_trace(
            go.Scattermapbox(
                lat=az_lat,
                lon=az_lon,
                mode="lines",
                line=dict(width=4, color=color),
                name=f"{dataset} azimuth",
                hoverinfo="skip",
            )
        )

        tilt_lat, tilt_lon = [], []
        for p in subset:
            lat0, lon0 = enu_to_latlon(p.E, p.N, UCSB_REF_LAT_DEG, UCSB_REF_LON_DEG)
            dE, dN, _ = tilt_dir(p.azimuth_deg, p.tilt_deg)
            lat1, lon1 = enu_to_latlon(
                p.E + dE * TILT_GROUND_LEN_M,
                p.N + dN * TILT_GROUND_LEN_M,
                UCSB_REF_LAT_DEG,
                UCSB_REF_LON_DEG,
            )
            tilt_lat += [lat0, lat1, None]
            tilt_lon += [lon0, lon1, None]

        fig.add_trace(
            go.Scattermapbox(
                lat=tilt_lat,
                lon=tilt_lon,
                mode="lines",
                line=dict(width=3, color=color),
                name=f"{dataset} tilt (ground proj)",
                opacity=0.55,
                hoverinfo="skip",
            )
        )

    # DARK MODE + set bounds so it starts at the data, not UCSB ref
    b = bounds_from_points(all_lat_lists, all_lon_lists, pad_frac=0.08)

    mapbox_layout: Dict[str, Any] = dict(
        style="carto-darkmatter",
    )
    if b is not None:
        mapbox_layout["bounds"] = b
    else:
        mapbox_layout["center"] = dict(lat=UCSB_REF_LAT_DEG, lon=UCSB_REF_LON_DEG)
        mapbox_layout["zoom"] = 16

    fig.update_layout(
        title=dict(text="All Pose Datasets + PLY (Dark)", font=dict(color="white")),
        mapbox=mapbox_layout,
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(orientation="h", font=dict(color="white")),
        paper_bgcolor="#111111",
        font=dict(color="white"),
    )

    fig.write_html(out_html, include_plotlyjs=True, config={"scrollZoom": True})
    print(f"Wrote: {out_html}")


if __name__ == "__main__":
    import sys

    # Usage:
    #   python multi_pose_view_2d_map.py <root_pose_dir> <out_html> [ply_path]
    root = sys.argv[1]
    out_html = sys.argv[2]
    ply_path = sys.argv[3] if len(sys.argv) >= 4 else None

    build_map_view(root, out_html, ply_path)