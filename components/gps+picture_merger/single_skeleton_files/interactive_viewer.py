from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import plotly.graph_objects as go
import yaml


@dataclass(frozen=True)
class PoseItem:
    name: int
    E: float
    N: float
    U: float
    tilt_deg: float
    azimuth_deg: float


class Pose2VecWithPlanesViewer:
    """
    Plots:
      - pose points (ENU)
      - azimuth vector (horizontal)
      - tilt vector (aligned with azimuth, uses tilt_deg)
      - a small horizontal plane (E-N plane) intersecting each point (at that point's U)

    ENU conventions:
      azimuth_deg: 0 -> +N, 90 -> +E

    Tilt convention USED HERE (matches what you just said):
      tilt_deg = 0    -> horizontal
      tilt_deg > 0    -> pitches UP (positive U)
      tilt_deg < 0    -> pitches DOWN (negative U)

    So the direction aligned with azimuth is:
      horiz = cos(tilt)
      dE = sin(az) * horiz
      dN = cos(az) * horiz
      dU = +sin(tilt)     # <-- POSITIVE means UP (this is the change you asked for)
    """

    def __init__(
        self,
        poses_yaml: str | Path,
        out_html: str | Path,
        *,
        az_len_m: float = 2.0,
        tilt_len_m: float = 3.5,
        plane_size_m: float = 0.8,   # square side length (meters) for each plane
        point_size: int = 6,
        line_width: int = 10,
        show_grid: bool = True,
    ) -> None:
        self.poses_yaml = Path(poses_yaml)
        self.out_html = Path(out_html)
        self.out_html.parent.mkdir(parents=True, exist_ok=True)

        self.az_len_m = float(az_len_m)
        self.tilt_len_m = float(tilt_len_m)
        self.plane_size_m = float(plane_size_m)
        self.point_size = int(point_size)
        self.line_width = int(line_width)
        self.show_grid = bool(show_grid)

    def build(self) -> Path:
        data = yaml.safe_load(self.poses_yaml.read_text(encoding="utf-8")) or {}
        poses = self._load_poses(data)
        if not poses:
            raise RuntimeError("No poses found in YAML.")

        # Points
        points = go.Scatter3d(
            x=[p.E for p in poses],
            y=[p.N for p in poses],
            z=[p.U for p in poses],
            mode="markers+text",
            text=[str(p.name) for p in poses],
            textposition="top center",
            marker=dict(size=self.point_size),
            name="poses",
        )

        # Vectors as line-collections (one trace each, separated by None)
        az_x: List[float] = []
        az_y: List[float] = []
        az_z: List[float] = []

        tilt_x: List[float] = []
        tilt_y: List[float] = []
        tilt_z: List[float] = []

        plane_traces: List[go.BaseTraceType] = []

        for p in poses:
            # (1) azimuth-only (horizontal)
            aE, aN, aU = self._azimuth_dir(p.azimuth_deg)
            az_x += [p.E, p.E + aE * self.az_len_m, None]
            az_y += [p.N, p.N + aN * self.az_len_m, None]
            az_z += [p.U, p.U + aU * self.az_len_m, None]

            # (2) tilt vector (THIS uses tilt_deg; positive tilt points UP)
            tE, tN, tU = self._tilt_dir_along_azimuth_up_is_positive(
                azimuth_deg=p.azimuth_deg,
                tilt_deg=p.tilt_deg,
            )
            tilt_x += [p.E, p.E + tE * self.tilt_len_m, None]
            tilt_y += [p.N, p.N + tN * self.tilt_len_m, None]
            tilt_z += [p.U, p.U + tU * self.tilt_len_m, None]

            # (3) plane intersecting the dot (a small E-N plane patch at z=U)
            plane_traces.append(self._plane_patch_at_point(p.E, p.N, p.U, size=self.plane_size_m))

        az_trace = go.Scatter3d(
            x=az_x, y=az_y, z=az_z,
            mode="lines",
            line=dict(width=self.line_width),
            name="azimuth (horizontal)",
            hoverinfo="skip",
        )

        tilt_trace = go.Scatter3d(
            x=tilt_x, y=tilt_y, z=tilt_z,
            mode="lines",
            line=dict(width=self.line_width),
            name="tilt (aligned w/ az, +tilt = up)",
            hoverinfo="skip",
        )

        fig = go.Figure(data=[points, az_trace, tilt_trace, *plane_traces])
        fig.update_layout(
            title="ENU poses — azimuth + tilt vectors (+tilt=up) + per-point planes",
            scene=dict(
                xaxis=dict(title="E (m)", showgrid=self.show_grid),
                yaxis=dict(title="N (m)", showgrid=self.show_grid),
                zaxis=dict(title="U (m)", showgrid=self.show_grid),
                aspectmode="data",
            ),
            margin=dict(l=0, r=0, t=50, b=0),
        )

        fig.write_html(self.out_html, include_plotlyjs=True, full_html=True)
        return self.out_html

    # -------------------------
    # VECTOR MATH
    # -------------------------
    def _azimuth_dir(self, azimuth_deg: float) -> Tuple[float, float, float]:
        """Azimuth direction in EN plane (unit), U=0."""
        az = math.radians(azimuth_deg)
        dE = math.sin(az)
        dN = math.cos(az)
        return self._unit(dE, dN, 0.0)

    # ==========================================================
    # *** TILT FUNCTION (MARKED) ***
    #
    # tilt_deg is treated as pitch relative to HORIZONTAL:
    #   tilt=0   -> horizontal
    #   tilt>0   -> points UP   (+U)
    #   tilt<0   -> points DOWN (-U)
    #
    # Combine azimuth forward direction f=(sin(az), cos(az), 0) with pitch:
    #   horizontal magnitude = cos(tilt)
    #   vertical component   = +sin(tilt)   <-- + means UP (your request)
    #
    # So:
    #   dE = sin(az) * cos(tilt)
    #   dN = cos(az) * cos(tilt)
    #   dU = +sin(tilt)
    # ==========================================================
    def _tilt_dir_along_azimuth_up_is_positive(self, azimuth_deg: float, tilt_deg: float) -> Tuple[float, float, float]:
        az = math.radians(azimuth_deg)
        tilt = math.radians(tilt_deg)

        horiz = math.cos(tilt)
        dE = math.sin(az) * horiz
        dN = math.cos(az) * horiz
        dU = math.sin(tilt)  # <-- flipped sign so +tilt points UP

        return self._unit(dE, dN, dU)

    def _unit(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        n = math.sqrt(x * x + y * y + z * z)
        if n <= 0:
            return 0.0, 0.0, 0.0
        return x / n, y / n, z / n

    # -------------------------
    # PLANE PATCH (per-point)
    # -------------------------
    def _plane_patch_at_point(self, E: float, N: float, U: float, *, size: float) -> go.Mesh3d:
        """
        A small square patch in the E-N plane (horizontal), centered at (E,N,U).
        Built as 2 triangles (Mesh3d).
        """
        s = size / 2.0

        # corners (counter-clockwise)
        x0, y0, z0 = E - s, N - s, U
        x1, y1, z1 = E + s, N - s, U
        x2, y2, z2 = E + s, N + s, U
        x3, y3, z3 = E - s, N + s, U

        # 4 vertices
        xs = [x0, x1, x2, x3]
        ys = [y0, y1, y2, y3]
        zs = [z0, z1, z2, z3]

        # two triangles: (0,1,2) and (0,2,3)
        i = [0, 0]
        j = [1, 2]
        k = [2, 3]

        return go.Mesh3d(
            x=xs, y=ys, z=zs,
            i=i, j=j, k=k,
            opacity=0.25,
            name="plane",
            hoverinfo="skip",
            showscale=False,
        )

    # -------------------------
    # YAML
    # -------------------------
    def _load_poses(self, data: Dict[str, Any]) -> List[PoseItem]:
        poses = data.get("poses", []) or []
        out: List[PoseItem] = []
        for p in poses:
            enu = p.get("enu_m") or {}
            out.append(
                PoseItem(
                    name=int(p["name"]),
                    E=float(enu.get("E", 0.0)),
                    N=float(enu.get("N", 0.0)),
                    U=float(enu.get("U", 0.0)),
                    tilt_deg=float(p.get("tilt_deg", 0.0)),
                    azimuth_deg=float(p.get("azimuth_deg", 0.0)),
                )
            )
        return out


if __name__ == "__main__":
    import sys

    # Usage:
    #   python pose_2vec_planes.py poses.yaml out.html
    poses_yaml = sys.argv[1]
    out_html = sys.argv[2]

    viewer = Pose2VecWithPlanesViewer(
        poses_yaml=poses_yaml,
        out_html=out_html,
        az_len_m=2.0,
        tilt_len_m=5.0,     # make bigger if you want it super obvious
        plane_size_m=0.9,
        point_size=7,
        line_width=12,
        show_grid=True,
    )
    out = viewer.build()
    print(f"Wrote: {out}")
