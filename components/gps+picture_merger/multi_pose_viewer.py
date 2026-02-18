from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import plotly.graph_objects as go
import yaml


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
# Vector helpers
# ------------------------------------------------------------
def unit(x, y, z):
    n = math.sqrt(x * x + y * y + z * z)
    if n <= 0:
        return 0.0, 0.0, 0.0
    return x / n, y / n, z / n


def azimuth_dir(azimuth_deg):
    az = math.radians(azimuth_deg)
    return unit(math.sin(az), math.cos(az), 0.0)


def tilt_dir(azimuth_deg, tilt_deg):
    az = math.radians(azimuth_deg)
    tilt = math.radians(tilt_deg)

    horiz = math.cos(tilt)
    dE = math.sin(az) * horiz
    dN = math.cos(az) * horiz
    dU = math.sin(tilt)
    return unit(dE, dN, dU)


# ------------------------------------------------------------
# Viewer
# ------------------------------------------------------------
def build_view(root_dir: str | Path, out_html: str | Path):
    root = Path(root_dir)
    out_html = Path(out_html)

    poses = load_all_pose_yamls(root)

    # ----------------------------
    # GLOBAL CENTER (UCSB center)
    # ----------------------------

    datasets = sorted(set(p.dataset for p in poses))
    colors = ["red", "blue", "green", "orange", "purple", "cyan", "magenta", "brown"]

    fig = go.Figure()

    # ---- add center marker first (so it’s easy to see)
# ---- UCSB ENU origin (reference point)
    fig.add_trace(go.Scatter3d(
        x=[0.0], y=[0.0], z=[0.0],
        mode="markers+text",
        text=["UCSB_REF (ENU origin)"],
        textposition="top center",
        marker=dict(size=10, color="black", symbol="diamond"),
        name="UCSB reference"
    ))


    for idx, dataset in enumerate(datasets):
        subset = [p for p in poses if p.dataset == dataset]
        color = colors[idx % len(colors)]

        fig.add_trace(go.Scatter3d(
            x=[p.E for p in subset],
            y=[p.N for p in subset],
            z=[p.U for p in subset],
            mode="markers+text",
            text=[f"{dataset}:{p.name}" for p in subset],
            marker=dict(size=6, color=color),
            name=f"{dataset} poses"
        ))

        az_x, az_y, az_z = [], [], []
        tilt_x, tilt_y, tilt_z = [], [], []

        for p in subset:
            aE, aN, aU = azimuth_dir(p.azimuth_deg)
            az_x += [p.E, p.E + aE * 2.0, None]
            az_y += [p.N, p.N + aN * 2.0, None]
            az_z += [p.U, p.U + aU * 2.0, None]

            tE, tN, tU = tilt_dir(p.azimuth_deg, p.tilt_deg)
            tilt_x += [p.E, p.E + tE * 4.0, None]
            tilt_y += [p.N, p.N + tN * 4.0, None]
            tilt_z += [p.U, p.U + tU * 4.0, None]

        fig.add_trace(go.Scatter3d(
            x=az_x, y=az_y, z=az_z,
            mode="lines",
            line=dict(width=6, color=color),
            name=f"{dataset} azimuth"
        ))

        fig.add_trace(go.Scatter3d(
            x=tilt_x, y=tilt_y, z=tilt_z,
            mode="lines",
            line=dict(width=6, dash="dash", color=color),
            name=f"{dataset} tilt"
        ))

    fig.update_layout(
        title="All Pose Datasets (Combined)",
        scene=dict(
            xaxis=dict(title="E (m)"),
            yaxis=dict(title="N (m)"),
            zaxis=dict(title="U (m)"),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=50, b=0),
    )

    fig.write_html(out_html, include_plotlyjs=True)
    print(f"Wrote: {out_html}")

# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
if __name__ == "__main__":
    import sys

    root = sys.argv[1]     # e.g. pose_viz_out
    out_html = sys.argv[2] # e.g. all_poses.html

    build_view(root, out_html)
