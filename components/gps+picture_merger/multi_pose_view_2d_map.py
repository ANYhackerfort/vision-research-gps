from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import plotly.graph_objects as go
import yaml

# ============================================================
# ENU reference (your "UCSB center")
# ============================================================
UCSB_REF_LAT_DEG = 34.413963
UCSB_REF_LON_DEG = -119.848946

# Earth radius (meters) for local tangent-plane approximations
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
    """
    Local tangent-plane approximation good for campus-scale extents.
    E, N in meters relative to reference lat/lon.
    """
    lat0 = math.radians(ref_lat_deg)
    dlat = N_m / EARTH_R_M
    dlon = E_m / (EARTH_R_M * math.cos(lat0))
    lat = ref_lat_deg + math.degrees(dlat)
    lon = ref_lon_deg + math.degrees(dlon)
    return lat, lon


# ------------------------------------------------------------
# Map viewer (tile map base)
# ------------------------------------------------------------
def build_map_view(root_dir: str | Path, out_html: str | Path):
    root = Path(root_dir)
    out_html = Path(out_html)

    poses = load_all_pose_yamls(root)
    datasets = sorted(set(p.dataset for p in poses))

    colors = ["red", "blue", "green", "orange", "purple", "cyan", "magenta", "brown"]

    fig = go.Figure()

    # UCSB reference marker (ENU origin)
    fig.add_trace(go.Scattermapbox(
        lat=[UCSB_REF_LAT_DEG],
        lon=[UCSB_REF_LON_DEG],
        mode="markers+text",
        text=["UCSB_REF (ENU origin)"],
        textposition="top center",
        marker=dict(size=14, color="black"),
        name="UCSB reference",
        hovertext=[f"ENU origin<br>lat={UCSB_REF_LAT_DEG:.6f}, lon={UCSB_REF_LON_DEG:.6f}"],
        hoverinfo="text"
    ))

    # Configurable ray lengths (meters in ENU)
    AZ_LEN_M = 10.0          # azimuth arrow length on ground
    TILT_GROUND_LEN_M = 15.0 # tilt ray projected on ground (still in az direction)

    # Plot per dataset
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

        # Points
        fig.add_trace(go.Scattermapbox(
            lat=lats,
            lon=lons,
            mode="markers+text",
            text=labels,
            textposition="top right",
            marker=dict(size=9, color=color),
            name=f"{dataset} poses",
            hovertext=hovers,
            hoverinfo="text",
        ))

        # Azimuth lines (solid)
        az_lat, az_lon = [], []
        for p in subset:
            lat0, lon0 = enu_to_latlon(p.E, p.N, UCSB_REF_LAT_DEG, UCSB_REF_LON_DEG)
            dE, dN, _ = azimuth_dir(p.azimuth_deg)
            lat1, lon1 = enu_to_latlon(p.E + dE * AZ_LEN_M, p.N + dN * AZ_LEN_M, UCSB_REF_LAT_DEG, UCSB_REF_LON_DEG)

            az_lat += [lat0, lat1, None]
            az_lon += [lon0, lon1, None]

        fig.add_trace(go.Scattermapbox(
            lat=az_lat,
            lon=az_lon,
            mode="lines",
            line=dict(width=3, color=color),
            name=f"{dataset} azimuth",
            hoverinfo="skip",
        ))

        # Tilt projected on ground (semi-transparent line)
        # (Map is 2D, so we can’t show the vertical component; this shows direction on the ground.)
        tilt_lat, tilt_lon = [], []
        for p in subset:
            lat0, lon0 = enu_to_latlon(p.E, p.N, UCSB_REF_LAT_DEG, UCSB_REF_LON_DEG)
            dE, dN, _ = tilt_dir(p.azimuth_deg, p.tilt_deg)
            lat1, lon1 = enu_to_latlon(p.E + dE * TILT_GROUND_LEN_M, p.N + dN * TILT_GROUND_LEN_M, UCSB_REF_LAT_DEG, UCSB_REF_LON_DEG)

            tilt_lat += [lat0, lat1, None]
            tilt_lon += [lon0, lon1, None]

        fig.add_trace(go.Scattermapbox(
            lat=tilt_lat,
            lon=tilt_lon,
            mode="lines",
            line=dict(width=2, color=color),
            name=f"{dataset} tilt (ground proj)",
            opacity=0.45,
            hoverinfo="skip",
        ))

    # Center the map around UCSB ref
    fig.update_layout(
        title="All Pose Datasets (Map Overlay)",
        mapbox=dict(
            style="open-street-map",  # no token needed
            center=dict(lat=UCSB_REF_LAT_DEG, lon=UCSB_REF_LON_DEG),
            zoom=16,
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(orientation="h"),
    )

    fig.write_html(out_html, include_plotlyjs=True)
    print(f"Wrote: {out_html}")


# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
if __name__ == "__main__":
    import sys

    root = sys.argv[1]     # e.g. pose_viz_out
    out_html = sys.argv[2] # e.g. all_poses_map.html

    build_map_view(root, out_html)
