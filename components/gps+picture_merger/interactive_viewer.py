from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Tuple

import plotly.graph_objects as go
import yaml
from PIL import Image


@dataclass(frozen=True)
class PoseItem:
    name: int
    image_path: Path
    E: float
    N: float
    U: float
    tilt_deg: float
    azimuth_deg: float


class Pose3DViewer:
    """
    Interactive 3D viewer (HTML) for ENU poses + image hover previews.

    - Points are ENU (meters) w.r.t. a reference LLH that produced the YAML.
    - Draws:
        * Azimuth heading ray in the horizontal EN plane
        * "Unknown vertical direction" circle in (forward-from-azimuth, Up) plane
        * Reference point marker + local axis triad (E, N, U) from reference
        * Optional "normalized" axes: unit-length E/N/U triad at reference

    Expected YAML format:
      reference_llh:
        lat: ...
        lon: ...
        h_ellip_m: ...
      poses:
        - name: ...
          image_path: ...
          enu_m: {E: ..., N: ..., U: ...}
          tilt_deg: ...
          azimuth_deg: ...
    """

    def __init__(
        self,
        poses_yaml: str | Path,
        images_root: str | Path,
        out_dir: str | Path,
        *,
        thumb_max_px: int = 220,
        heading_len_m: float = 1.0,
        circle_radius_m: float = 0.8,
        circle_segments: int = 40,
        axis_len_m: float = 2.0,
        show_normalized_axes: bool = True,
    ) -> None:
        self.poses_yaml = Path(poses_yaml)
        self.images_root = Path(images_root)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.thumb_max_px = int(thumb_max_px)
        self.heading_len_m = float(heading_len_m)
        self.circle_radius_m = float(circle_radius_m)
        self.circle_segments = int(circle_segments)
        self.axis_len_m = float(axis_len_m)
        self.show_normalized_axes = bool(show_normalized_axes)

    def build(self) -> Path:
        data = yaml.safe_load(self.poses_yaml.read_text(encoding="utf-8")) or {}
        poses = self._load_poses(data)
        if not poses:
            raise RuntimeError("No poses found in poses YAML.")

        # Reference point in ENU should be (0,0,0) if YAML was generated that way.
        # We'll plot it explicitly as the origin.
        ref_llh = data.get("reference_llh") or {}
        ref_info = {
            "lat": ref_llh.get("lat", None),
            "lon": ref_llh.get("lon", None),
            "h_ellip_m": ref_llh.get("h_ellip_m", None),
        }
        ref_E, ref_N, ref_U = 0.0, 0.0, 0.0

        # Points
        xs = [p.E for p in poses]
        ys = [p.N for p in poses]
        zs = [p.U for p in poses]

        # Hover HTML (thumbnail + info)
        hover_html: List[str] = []
        for p in poses:
            img_abs = self._resolve_image_path(p.image_path)
            thumb_data_uri = self._image_to_data_uri(img_abs)

            hover_html.append(
                f"""
                <div style="width:{self.thumb_max_px+50}px">
                  <div><b>Name:</b> {p.name}</div>
                  <div><b>ENU (m):</b> E={p.E:.3f}, N={p.N:.3f}, U={p.U:.3f}</div>
                  <div><b>Azimuth (deg):</b> {p.azimuth_deg:.1f}</div>
                  <div><b>Tilt (deg):</b> {p.tilt_deg:.1f}</div>
                  <div style="margin-top:6px;">
                    <img src="{thumb_data_uri}" style="max-width:{self.thumb_max_px}px; border-radius:10px;"/>
                  </div>
                  <div style="margin-top:6px; font-size:11px; color:#666;">
                    {img_abs.name}
                  </div>
                </div>
                """
            )

        points_trace = go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="markers+text",
            text=[str(p.name) for p in poses],
            textposition="top center",
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover_html,
            marker=dict(size=5),
            name="poses",
        )

        # -----------------------------
        # Heading rays from azimuth only (horizontal)
        # ENU convention:
        #   azimuth = 0 -> North (+N)
        #   azimuth = 90 -> East (+E)
        # so:
        #   dE = sin(az), dN = cos(az), dU = 0
        # -----------------------------
        ray_x: List[float] = []
        ray_y: List[float] = []
        ray_z: List[float] = []

        for p in poses:
            az = math.radians(p.azimuth_deg)
            dE = math.sin(az)
            dN = math.cos(az)
            dU = 0.0

            ray_x += [p.E, p.E + dE * self.heading_len_m, None]
            ray_y += [p.N, p.N + dN * self.heading_len_m, None]
            ray_z += [p.U, p.U + dU * self.heading_len_m, None]

        heading_trace = go.Scatter3d(
            x=ray_x,
            y=ray_y,
            z=ray_z,
            mode="lines",
            hoverinfo="skip",
            name="azimuth heading",
        )

        # -----------------------------
        # "Unknown vertical direction" circle:
        # Circle in plane spanned by:
        #   forward = (sin az, cos az, 0)
        #   up      = (0, 0, 1)
        # centered at the pose.
        # -----------------------------
        circ_x: List[float] = []
        circ_y: List[float] = []
        circ_z: List[float] = []

        for p in poses:
            az = math.radians(p.azimuth_deg)
            fE, fN, fU = math.sin(az), math.cos(az), 0.0
            uE, uN, uU = 0.0, 0.0, 1.0

            for k in range(self.circle_segments + 1):
                t = 2.0 * math.pi * (k / self.circle_segments)
                vE = math.cos(t) * fE + math.sin(t) * uE
                vN = math.cos(t) * fN + math.sin(t) * uN
                vU = math.cos(t) * fU + math.sin(t) * uU

                circ_x.append(p.E + self.circle_radius_m * vE)
                circ_y.append(p.N + self.circle_radius_m * vN)
                circ_z.append(p.U + self.circle_radius_m * vU)

            circ_x.append(None)
            circ_y.append(None)
            circ_z.append(None)

        circle_trace = go.Scatter3d(
            x=circ_x,
            y=circ_y,
            z=circ_z,
            mode="lines",
            hoverinfo="skip",
            name="unknown vertical (circle)",
        )

        # -----------------------------
        # Reference point marker (origin) + axis triad from reference
        # You asked: "POINT OUT where the reference point is and draw the west east north vector"
        #
        # In ENU:
        #   +E is East, -E is West
        #   +N is North, -N is South
        #   +U is Up,   -U is Down
        #
        # We'll draw:
        #   East arrow (+E), West arrow (-E), North arrow (+N), and Up arrow (+U)
        # from the reference point (0,0,0).
        # -----------------------------
        ref_hover = (
            f"<div><b>REFERENCE POINT</b></div>"
            f"<div>ENU origin: E=0, N=0, U=0</div>"
            f"<div style='margin-top:6px; font-size:11px; color:#666;'>"
            f"LLH: lat={ref_info['lat']}, lon={ref_info['lon']}, h={ref_info['h_ellip_m']}"
            f"</div>"
        )

        ref_point_trace = go.Scatter3d(
            x=[ref_E],
            y=[ref_N],
            z=[ref_U],
            mode="markers+text",
            text=["REF"],
            textposition="top center",
            hovertemplate=f"{ref_hover}<extra></extra>",
            marker=dict(size=7),
            name="reference",
        )

        axis_len = self.axis_len_m

        # Axis line segments (each segment separated by None)
        # East (+E)
        ax_x = [ref_E, ref_E + axis_len, None]
        ax_y = [ref_N, ref_N, None]
        ax_z = [ref_U, ref_U, None]
        # West (-E)
        ax_x += [ref_E, ref_E - axis_len, None]
        ax_y += [ref_N, ref_N, None]
        ax_z += [ref_U, ref_U, None]
        # North (+N)
        ax_x += [ref_E, ref_E, None]
        ax_y += [ref_N, ref_N + axis_len, None]
        ax_z += [ref_U, ref_U, None]
        # Up (+U)
        ax_x += [ref_E, ref_E, None]
        ax_y += [ref_N, ref_N, None]
        ax_z += [ref_U, ref_U + axis_len, None]

        axes_trace = go.Scatter3d(
            x=ax_x,
            y=ax_y,
            z=ax_z,
            mode="lines",
            hoverinfo="skip",
            name="REF axes (E/W/N/U)",
        )

        # Add small text labels at the ends (optional but helpful)
        labels_trace = go.Scatter3d(
            x=[ref_E + axis_len, ref_E - axis_len, ref_E, ref_E, ref_E],
            y=[ref_N, ref_N, ref_N + axis_len, ref_N, ref_N],
            z=[ref_U, ref_U, ref_U, ref_U + axis_len, ref_U],
            mode="text",
            text=["E", "W", "N", "U", ""],
            hoverinfo="skip",
            name="axis labels",
        )

        # Normalized coordinate system (unit axes) from reference
        # This is basically the same as above but length=1 (unit vectors).
        traces = [points_trace, heading_trace, circle_trace, ref_point_trace, axes_trace, labels_trace]
        if self.show_normalized_axes:
            unit = 1.0
            nx = [ref_E, ref_E + unit, None, ref_E, ref_E, None, ref_E, ref_E, None]
            ny = [ref_N, ref_N, None, ref_N, ref_N + unit, None, ref_N, ref_N, None]
            nz = [ref_U, ref_U, None, ref_U, ref_U, None, ref_U, ref_U + unit, None]

            norm_axes_trace = go.Scatter3d(
                x=nx,
                y=ny,
                z=nz,
                mode="lines",
                hoverinfo="skip",
                name="normalized axes (unit E/N/U)",
            )
            traces.append(norm_axes_trace)

        fig = go.Figure(data=traces)
        fig.update_layout(
            title="Pose viewer (ENU) — azimuth rays + unknown-vertical circles + reference axes",
            scene=dict(
                xaxis_title="E (m)",
                yaxis_title="N (m)",
                zaxis_title="U (m)",
                aspectmode="data",
            ),
            margin=dict(l=0, r=0, t=50, b=0),
        )

        out_html = self.out_dir / "viewer.html"
        fig.write_html(out_html, include_plotlyjs=True, full_html=True)
        return out_html

    # -------------------------
    # Internals
    # -------------------------
    def _load_poses(self, data: Dict[str, Any]) -> List[PoseItem]:
        poses = data.get("poses", []) or []
        out: List[PoseItem] = []

        for p in poses:
            enu = p.get("enu_m") or {}
            out.append(
                PoseItem(
                    name=int(p["name"]),
                    image_path=Path(p["image_path"]),
                    E=float(enu["E"]),
                    N=float(enu["N"]),
                    U=float(enu["U"]),
                    tilt_deg=float(p.get("tilt_deg", 0.0)),
                    azimuth_deg=float(p.get("azimuth_deg", 0.0)),
                )
            )

        return out

    def _resolve_image_path(self, image_path: Path) -> Path:
        if image_path.is_absolute():
            return image_path

        candidate = (self.images_root / image_path).resolve()
        if candidate.exists():
            return candidate

        candidate2 = (self.poses_yaml.parent / image_path).resolve()
        if candidate2.exists():
            return candidate2

        candidate3 = (self.images_root / image_path.name).resolve()
        return candidate3

    def _image_to_data_uri(self, img_path: Path) -> str:
        img = Image.open(img_path).convert("RGB")
        img.thumbnail((self.thumb_max_px, self.thumb_max_px))

        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"


if __name__ == "__main__":
    import sys

    # Usage:
    #   python pose_viewer.py poses.yaml . ./pose_viz_out
    #
    poses_yaml = sys.argv[1]      # e.g. ./output/poses.yaml
    images_root = sys.argv[2]     # e.g. . (project root)
    out_dir = sys.argv[3]         # e.g. ./pose_viz_out

    viewer = Pose3DViewer(
        poses_yaml,
        images_root,
        out_dir,
        thumb_max_px=220,
        heading_len_m=1.0,
        circle_radius_m=0.8,
        circle_segments=40,
        axis_len_m=2.0,
        show_normalized_axes=True,
    )
    out = viewer.build()
    print(f"Wrote: {out}")