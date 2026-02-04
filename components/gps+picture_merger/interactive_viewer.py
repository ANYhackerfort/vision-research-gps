from __future__ import annotations

import base64
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
    dE: float
    dN: float
    dU: float


class Pose3DViewer:
    """
    Builds an interactive 3D viewer (HTML) for ENU poses + image hover previews.

    Inputs:
      - poses_yaml: path to poses_enu.yaml (from your GPSPosePlotter)
      - images_root: base directory to resolve relative image paths
      - out_dir: output directory

    Output:
      - out_dir/viewer.html
    """

    def __init__(
        self,
        poses_yaml: str | Path,
        images_root: str | Path,
        out_dir: str | Path,
        thumb_max_px: int = 220,
        arrow_scale_m: float = 1.0,
    ) -> None:
        self.poses_yaml = Path(poses_yaml)
        self.images_root = Path(images_root)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.thumb_max_px = int(thumb_max_px)
        self.arrow_scale_m = float(arrow_scale_m)

    def build(self) -> Path:
        poses = self._load_poses()
        if not poses:
            raise RuntimeError("No poses found in poses YAML.")

        # Points
        xs = [p.E for p in poses]
        ys = [p.N for p in poses]
        zs = [p.U for p in poses]

        # Build hover HTML (thumbnail + info)
        hover_html: List[str] = []
        for p in poses:
            img_abs = self._resolve_image_path(p.image_path)
            thumb_data_uri = self._image_to_data_uri(img_abs)

            hover_html.append(
                f"""
                <div style="width:{self.thumb_max_px+40}px">
                  <div><b>Name:</b> {p.name}</div>
                  <div><b>ENU (m):</b> E={p.E:.3f}, N={p.N:.3f}, U={p.U:.3f}</div>
                  <div><b>Tilt (deg):</b> {p.tilt_deg:.2f}</div>
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

        # Orientation arrows as 3D line segments (simple + robust)
        arrow_x: List[float] = []
        arrow_y: List[float] = []
        arrow_z: List[float] = []

        for p in poses:
            arrow_x += [p.E, p.E + p.dE * self.arrow_scale_m, None]
            arrow_y += [p.N, p.N + p.dN * self.arrow_scale_m, None]
            arrow_z += [p.U, p.U + p.dU * self.arrow_scale_m, None]

        arrows_trace = go.Scatter3d(
            x=arrow_x,
            y=arrow_y,
            z=arrow_z,
            mode="lines",
            hoverinfo="skip",
            name="tilt rays",
        )

        fig = go.Figure(data=[points_trace, arrows_trace])
        fig.update_layout(
            title="Pose viewer (ENU) — hover points to see images",
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
    def _load_poses(self) -> List[PoseItem]:
        data = yaml.safe_load(self.poses_yaml.read_text(encoding="utf-8"))
        poses = data.get("poses", [])
        out: List[PoseItem] = []

        for p in poses:
            name = int(p["name"])
            image = Path(p["image"])
            out.append(
                PoseItem(
                    name=name,
                    image_path=image,
                    E=float(p["E_m"]),
                    N=float(p["N_m"]),
                    U=float(p["U_m"]),
                    tilt_deg=float(p.get("tilt_deg", 0.0)),
                    dE=float(p["dir_ENU"]["dE"]),
                    dN=float(p["dir_ENU"]["dN"]),
                    dU=float(p["dir_ENU"]["dU"]),
                )
            )

        return out

    def _resolve_image_path(self, image_path: Path) -> Path:
        # If YAML stored a relative path like "output/images_tiff/IMG_2245.tiff",
        # resolve it relative to images_root’s parent and cwd-like behavior.
        if image_path.is_absolute():
            return image_path
        # Try relative to project root passed in
        candidate = (self.images_root / image_path).resolve()
        if candidate.exists():
            return candidate
        # Try relative to poses_yaml directory
        candidate2 = (self.poses_yaml.parent / image_path).resolve()
        if candidate2.exists():
            return candidate2
        # As a final fallback, treat it as relative to images_root directly
        candidate3 = (self.images_root / image_path.name).resolve()
        return candidate3

    def _image_to_data_uri(self, img_path: Path) -> str:
        # Make a small JPEG thumbnail, embed as data URI for hover display
        img = Image.open(img_path).convert("RGB")
        img.thumbnail((self.thumb_max_px, self.thumb_max_px))

        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"


if __name__ == "__main__":
    import sys

    poses_yaml = sys.argv[1]      # e.g. ./pose_viz_out/poses_enu.yaml
    images_root = sys.argv[2]     # e.g. . (project root) OR ./output/images_tiff
    out_dir = sys.argv[3]         # e.g. ./pose_viz_out

    viewer = Pose3DViewer(poses_yaml, images_root, out_dir, thumb_max_px=220, arrow_scale_m=1.0)
    out = viewer.build()
    print(f"Wrote: {out}")
