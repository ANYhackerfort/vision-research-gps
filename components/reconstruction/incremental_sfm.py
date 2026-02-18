from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import h5py
import numpy as np


# ============================================================
# Intrinsics from assumed FOV (fallback when intrinsics unknown)
# ============================================================
def K_from_fov(image_size_hw: Tuple[int, int], fov_deg: float) -> np.ndarray:
    h, w = image_size_hw
    fov = math.radians(float(fov_deg))
    fx = 0.5 * w / math.tan(0.5 * fov)
    fy = fx
    cx = 0.5 * w
    cy = 0.5 * h
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


# ============================================================
# PLY writers
# ============================================================
def write_ply_xyz(path: Path, xyz: np.ndarray) -> None:
    xyz = np.asarray(xyz, dtype=np.float64)
    with path.open("w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {xyz.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for p in xyz:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")


def write_ply_xyzrgb(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    xyz = np.asarray(xyz, dtype=np.float64)
    rgb = np.asarray(rgb, dtype=np.uint8)
    assert xyz.shape[0] == rgb.shape[0]

    with path.open("w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {xyz.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(xyz, rgb):
            f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")


# ============================================================
# Robust TIFF-ish image reader (returns BGR uint8)
# ============================================================
def read_color_u8(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read image for color: {path}")

    # grayscale -> BGR
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # 16-bit -> 8-bit
    if img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)

    # float -> 8-bit
    if img.dtype in (np.float32, np.float64):
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)

    # Some TIFF readers return BGRA
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    return img  # BGR uint8


# ============================================================
# Track DB (union-find)
# ============================================================
@dataclass
class Obs:
    img: str
    kidx: int


class TrackDB:
    """
    Track = a 3D point + list of 2D observations (img, keypoint index).

    We build tracks by unioning correspondences for verified inlier pairs.
    """

    def __init__(self) -> None:
        self.parent: Dict[Tuple[str, int], Tuple[str, int]] = {}
        self.rank: Dict[Tuple[str, int], int] = {}
        self.members: Dict[Tuple[str, int], List[Obs]] = {}  # root -> obs list

    def _find(self, x: Tuple[str, int]) -> Tuple[str, int]:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            self.members[x] = [Obs(x[0], x[1])]
            return x
        if self.parent[x] != x:
            self.parent[x] = self._find(self.parent[x])
        return self.parent[x]

    def _union(self, a: Tuple[str, int], b: Tuple[str, int]) -> None:
        ra = self._find(a)
        rb = self._find(b)
        if ra == rb:
            return

        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra

        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

        self.members[ra].extend(self.members[rb])
        del self.members[rb]

    def add_match(self, img0: str, idx0: int, img1: str, idx1: int) -> None:
        self._union((img0, int(idx0)), (img1, int(idx1)))

    def observations(self, root: Tuple[str, int]) -> List[Obs]:
        return self.members[root]


# ============================================================
# Incremental SfM (minimal, intrinsics assumed via FOV)
# ============================================================
class IncrementalSfM:
    """
    Minimal incremental SfM pipeline:
      1) Load keypoints + image sizes from features_h5
      2) Load verified inlier pairs and build tracks (union-find)
      3) Pick seed pair (max inliers), recover pose (Essential, with assumed intrinsics)
      4) Triangulate seed points
      5) Incrementally add images via PnP from 2D-3D correspondences, triangulate new points
      6) Export points.ply, points_rgb.ply, cameras.json

    Notes:
      - Intrinsics are guessed from image size + assumed horizontal FOV.
      - Output geometry is up-to-scale / approximate, but good enough to bootstrap.
    """

    def __init__(
        self,
        features_h5: Path,
        verified_matches_h5: Path,
        out_dir: Path,
        images_dir: Optional[Path] = None,
        fov_deg: float = 60.0,
        seed_min_inliers: int = 80,
        pnp_min_inliers: int = 40,
        pnp_reproj_thresh_px: float = 4.0,
        max_images: Optional[int] = None,
    ) -> None:
        self.features_h5 = Path(features_h5)
        self.verified_matches_h5 = Path(verified_matches_h5)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Where to read original images (for coloring the cloud)
        if images_dir is None:
            # features_h5: .../outputs/features/<dataset>/superpoint_features.h5
            # parents[2] -> .../outputs
            outputs_dir = self.features_h5.parents[2]
            self.images_dir = (outputs_dir / "normalized" / "images").resolve()
        else:
            self.images_dir = Path(images_dir).resolve()

        self.fov_deg = float(fov_deg)
        self.seed_min_inliers = int(seed_min_inliers)
        self.pnp_min_inliers = int(pnp_min_inliers)
        self.pnp_reproj_thresh_px = float(pnp_reproj_thresh_px)
        self.max_images = max_images

        # Loaded from features
        self.K: Dict[str, np.ndarray] = {}
        self.kpts: Dict[str, np.ndarray] = {}
        self.imsize: Dict[str, Tuple[int, int]] = {}  # (h,w)

        # Camera extrinsics: X_cam = R * X_world + t
        self.R: Dict[str, np.ndarray] = {}
        self.t: Dict[str, np.ndarray] = {}

        # Tracks + 3D
        self.tracks = TrackDB()
        self.track_xyz: Dict[Tuple[str, int], np.ndarray] = {}  # track root -> xyz

        # Cache images for coloring (avoid re-reading same TIFF constantly)
        self._img_cache: Dict[str, np.ndarray] = {}

    # -----------------------------
    # Loading
    # -----------------------------
    def _load_features(self) -> None:
        with h5py.File(self.features_h5, "r") as f:
            for img in f.keys():
                g = f[img]
                k = np.array(g["keypoints"], dtype=np.float64)  # [N,2] (x,y)
                sz = np.array(g["image_size"], dtype=np.int32)  # [2] (h,w)
                h, w = int(sz[0]), int(sz[1])

                self.kpts[img] = k
                self.imsize[img] = (h, w)
                self.K[img] = K_from_fov((h, w), self.fov_deg)

    def _load_verified_edges_build_tracks(self) -> List[Tuple[str, str, np.ndarray, np.ndarray]]:
        edges: List[Tuple[str, str, np.ndarray, np.ndarray]] = []
        with h5py.File(self.verified_matches_h5, "r") as m:
            pairs = m.get("pairs", None)
            if pairs is None:
                raise RuntimeError("verified_matches_h5 missing group: /pairs")

            for pair_key in pairs.keys():
                g = pairs[pair_key]
                img0 = str(g.attrs["img0"])
                img1 = str(g.attrs["img1"])
                idx0 = np.array(g["inlier_idx0"], dtype=np.int32)
                idx1 = np.array(g["inlier_idx1"], dtype=np.int32)
                if idx0.size == 0:
                    continue

                edges.append((img0, img1, idx0, idx1))

                # Union into tracks
                for a, b in zip(idx0, idx1):
                    self.tracks.add_match(img0, int(a), img1, int(b))

        return edges

    # -----------------------------
    # Seed selection
    # -----------------------------
    def _pick_seed_pair(self, edges: List[Tuple[str, str, np.ndarray, np.ndarray]]) -> Tuple[str, str, np.ndarray, np.ndarray]:
        edges_sorted = sorted(edges, key=lambda e: e[2].size, reverse=True)
        for img0, img1, idx0, idx1 in edges_sorted:
            if idx0.size >= self.seed_min_inliers:
                return img0, img1, idx0, idx1
        return edges_sorted[0]

    # -----------------------------
    # Two-view pose + triangulation
    # -----------------------------
    def _recover_pose_seed(self, img0: str, img1: str, idx0: np.ndarray, idx1: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        K0 = self.K[img0]
        pts0 = self.kpts[img0][idx0]
        pts1 = self.kpts[img1][idx1]

        E, mask = cv2.findEssentialMat(
            pts0, pts1,
            cameraMatrix=K0,
            method=getattr(cv2, "USAC_MAGSAC", cv2.RANSAC),
            prob=0.999,
            threshold=2.0,
            maxIters=5000,
        )
        if E is None or mask is None:
            raise RuntimeError("findEssentialMat failed on seed pair")

        mask = mask.reshape(-1).astype(bool)
        pts0_in = pts0[mask]
        pts1_in = pts1[mask]
        idx0_in = idx0[mask]
        idx1_in = idx1[mask]

        _, R, t, mask_pose = cv2.recoverPose(E, pts0_in, pts1_in, K0)
        if mask_pose is None:
            raise RuntimeError("recoverPose failed on seed pair")

        mask_pose = mask_pose.reshape(-1).astype(bool)
        return R, t.reshape(3, 1), np.vstack([idx0_in[mask_pose], idx1_in[mask_pose]])

    def _triangulate_pair(self, img0: str, img1: str, idx0: np.ndarray, idx1: np.ndarray):
        K0 = np.asarray(self.K[img0], dtype=np.float64)
        K1 = np.asarray(self.K[img1], dtype=np.float64)
        R0 = np.asarray(self.R[img0], dtype=np.float64)
        t0 = np.asarray(self.t[img0], dtype=np.float64).reshape(3, 1)
        R1 = np.asarray(self.R[img1], dtype=np.float64)
        t1 = np.asarray(self.t[img1], dtype=np.float64).reshape(3, 1)

        P0 = (K0 @ np.hstack([R0, t0])).astype(np.float64)
        P1 = (K1 @ np.hstack([R1, t1])).astype(np.float64)

        pts0 = np.asarray(self.kpts[img0][idx0], dtype=np.float64)  # [N,2]
        pts1 = np.asarray(self.kpts[img1][idx1], dtype=np.float64)  # [N,2]

        # OpenCV expects 2xN contiguous float matrices
        pts0 = np.ascontiguousarray(pts0.T)  # [2,N]
        pts1 = np.ascontiguousarray(pts1.T)  # [2,N]

        if pts0.shape[1] < 2:
            return np.zeros((0, 3), dtype=np.float64), idx0[:0], idx1[:0]

        X_h = cv2.triangulatePoints(P0, P1, pts0, pts1)  # [4,N]
        X = (X_h[:3] / X_h[3:4]).T  # [N,3]

        # Positive depth in both cameras
        z0 = (R0 @ X.T + t0).T[:, 2]
        z1 = (R1 @ X.T + t1).T[:, 2]
        good = (z0 > 0.0) & (z1 > 0.0) & np.isfinite(X).all(axis=1)

        return X[good], idx0[good], idx1[good]


    def _triangulate_assign_tracks_for_pair(self, img0: str, img1: str, idx0: np.ndarray, idx1: np.ndarray) -> None:
        X, u0, u1 = self._triangulate_pair(img0, img1, idx0, idx1)
        for xyz, a, b in zip(X, u0, u1):
            root = self.tracks._find((img0, int(a)))
            if root not in self.track_xyz:
                self.track_xyz[root] = xyz

    # -----------------------------
    # PnP for next image
    # -----------------------------
    def _estimate_pose_from_tracks(self, img: str) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        obj = []
        imgpts = []

        # Build 2D-3D correspondences from existing tracks
        for root, xyz in self.track_xyz.items():
            for o in self.tracks.observations(root):
                if o.img == img:
                    obj.append(xyz)
                    imgpts.append(self.kpts[img][o.kidx])
                    break

        if len(obj) < self.pnp_min_inliers:
            return False, None, None

        obj_xyz = np.array(obj, dtype=np.float64)
        img_xy = np.array(imgpts, dtype=np.float64)

        K = self.K[img]
        dist = np.zeros((4, 1), dtype=np.float64)

        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=obj_xyz,
            imagePoints=img_xy,
            cameraMatrix=K,
            distCoeffs=dist,
            iterationsCount=5000,
            reprojectionError=self.pnp_reproj_thresh_px,
            confidence=0.999,
            flags=cv2.SOLVEPNP_EPNP,
        )

        if not ok or inliers is None or inliers.size < self.pnp_min_inliers:
            return False, None, None

        R, _ = cv2.Rodrigues(rvec)
        t = tvec.reshape(3, 1)
        return True, R, t

    def _count_2d3d(self, img: str) -> int:
        cnt = 0
        for root in self.track_xyz.keys():
            for o in self.tracks.observations(root):
                if o.img == img:
                    cnt += 1
                    break
        return cnt

    def _triangulate_new_points_for_image(
        self,
        img_new: str,
        edges: List[Tuple[str, str, np.ndarray, np.ndarray]],
    ) -> None:
        for img0, img1, idx0, idx1 in edges:
            if img0 == img_new and img1 in self.R:
                self._triangulate_assign_tracks_for_pair(img0, img1, idx0, idx1)
            elif img1 == img_new and img0 in self.R:
                self._triangulate_assign_tracks_for_pair(img1, img0, idx1, idx0)

    # -----------------------------
    # Coloring
    # -----------------------------
    def _get_image_cached(self, img_name: str) -> Optional[np.ndarray]:
        if img_name in self._img_cache:
            return self._img_cache[img_name]

        path = (self.images_dir / img_name)
        if not path.exists():
            return None

        im = read_color_u8(path)
        self._img_cache[img_name] = im
        return im

    def _color_for_track(self, root: Tuple[str, int], feats: h5py.File) -> np.ndarray:
        # Pick first observation that has an image file
        obs_list = self.tracks.observations(root)
        for o in obs_list:
            im = self._get_image_cached(o.img)
            if im is None:
                continue

            # keypoint location (x,y)
            kxy = np.array(feats[o.img]["keypoints"][o.kidx], dtype=np.float64)
            h, w = im.shape[:2]
            x = int(round(kxy[0]))
            y = int(round(kxy[1]))
            if x < 0:
                x = 0
            elif x >= w:
                x = w - 1
            if y < 0:
                y = 0
            elif y >= h:
                y = h - 1

            b, g, r = im[y, x]
            return np.array([r, g, b], dtype=np.uint8)

        return np.array([255, 255, 255], dtype=np.uint8)  # fallback: white

    # -----------------------------
    # Run
    # -----------------------------
    def run(self) -> None:
        self._load_features()
        edges = self._load_verified_edges_build_tracks()
        if not edges:
            raise RuntimeError("No verified edges found (verified_matches.h5 had no inliers).")

        seed0, seed1, sidx0, sidx1 = self._pick_seed_pair(edges)

        # Pose seed0 at origin
        self.R[seed0] = np.eye(3, dtype=np.float64)
        self.t[seed0] = np.zeros((3, 1), dtype=np.float64)

        # Recover pose for seed1
        R10, t10, idx_stacked = self._recover_pose_seed(seed0, seed1, sidx0, sidx1)
        self.R[seed1] = R10
        self.t[seed1] = t10

        posed: Set[str] = {seed0, seed1}

        # Triangulate seed points (only the pose-consistent subset)
        idx0_pose = idx_stacked[0].astype(np.int32)
        idx1_pose = idx_stacked[1].astype(np.int32)
        self._triangulate_assign_tracks_for_pair(seed0, seed1, idx0_pose, idx1_pose)

        # Incremental add
        all_images = sorted(self.kpts.keys())
        if self.max_images is not None:
            all_images = all_images[: int(self.max_images)]

        progress = True
        while progress:
            progress = False

            candidates = [img for img in all_images if img not in posed]
            scored = [(img, self._count_2d3d(img)) for img in candidates]
            scored.sort(key=lambda x: x[1], reverse=True)

            for img, cnt in scored:
                if cnt < self.pnp_min_inliers:
                    continue

                ok, R, t = self._estimate_pose_from_tracks(img)
                if not ok:
                    continue

                self.R[img] = R
                self.t[img] = t
                posed.add(img)
                progress = True

                # Triangulate more points with posed neighbors
                self._triangulate_new_points_for_image(img, edges)

                # Greedy: add one per loop, then rescore
                break

        # Export: XYZ + XYZRGB
        xyz_list: List[np.ndarray] = []
        rgb_list: List[np.ndarray] = []

        with h5py.File(self.features_h5, "r") as feats:
            for root, xyz in self.track_xyz.items():
                xyz_list.append(xyz)
                rgb_list.append(self._color_for_track(root, feats))

        xyz = np.array(xyz_list, dtype=np.float64)
        rgb = np.array(rgb_list, dtype=np.uint8)

        ply_xyz = self.out_dir / "points.ply"
        ply_rgb = self.out_dir / "points_rgb.ply"
        write_ply_xyz(ply_xyz, xyz)
        write_ply_xyzrgb(ply_rgb, xyz, rgb)

        # Export cameras
        cam_out = {}
        for img in sorted(posed):
            cam_out[img] = {
                "R": self.R[img].tolist(),
                "t": self.t[img].reshape(3).tolist(),
                "K": self.K[img].tolist(),
            }
        cams_json = self.out_dir / "cameras.json"
        cams_json.write_text(json.dumps(cam_out, indent=2))

        print(f"[OK] posed cameras: {len(posed)} / {len(all_images)}")
        print(f"[OK] points: {xyz.shape[0]}")
        print(f"[OK] wrote: {ply_xyz}")
        print(f"[OK] wrote: {ply_rgb}")
        print(f"[OK] wrote: {cams_json}")
        print(f"[OK] images_dir used for color: {self.images_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-h5", required=True)
    ap.add_argument("--verified-matches-h5", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--images-dir", default=None, help="Optional override: path to normalized/images folder (for coloring)")

    ap.add_argument("--fov-deg", type=float, default=60.0)
    ap.add_argument("--seed-min-inliers", type=int, default=80)
    ap.add_argument("--pnp-min-inliers", type=int, default=40)
    ap.add_argument("--pnp-reproj-thresh", type=float, default=4.0)
    ap.add_argument("--max-images", type=int, default=None)

    args = ap.parse_args()

    sfm = IncrementalSfM(
        features_h5=Path(args.features_h5),
        verified_matches_h5=Path(args.verified_matches_h5),
        out_dir=Path(args.out_dir),
        images_dir=Path(args.images_dir) if args.images_dir else None,
        fov_deg=args.fov_deg,
        seed_min_inliers=args.seed_min_inliers,
        pnp_min_inliers=args.pnp_min_inliers,
        pnp_reproj_thresh_px=args.pnp_reproj_thresh,
        max_images=args.max_images,
    )
    sfm.run()


if __name__ == "__main__":
    main()
