# ============================================================
# Campbell SfM pipeline (SuperPoint -> LightGlue -> RANSAC -> Incremental SfM)
# Assumes you already have:
#   outputs/normalized/poses_normalized.yaml
#   outputs/normalized/cameras.csv
#   outputs/normalized/images/   (canonical image names like Campbell_0001_....tiff)
# ============================================================

# --- set once (edit to your path) ---
ROOT="/home/linghe-zhang/vision-research/components/gps+picture_merger/pose_viz_out/outputs"
NORM="$ROOT/normalized"
DATASET="Campbell"

# (optional) sanity check paths
ls "$NORM/poses_normalized.yaml" "$NORM/cameras.csv" | cat
ls "$NORM/images" | head

# ============================================================
# 1) SuperPoint feature extraction (writes features/<DATASET>/superpoint_features.h5)
# ============================================================
python3 extract_superpoint.py \
  --normalized-root "$NORM" \
  --dataset "$DATASET" \
  --max-side 1600 \
  --max-kpts 2048 \
  --device cuda \
  --overwrite

# output:
#   $ROOT/features/Campbell/superpoint_features.h5


# ============================================================
# 2) LightGlue matching with ENU-KNN pairing (writes matches/<DATASET>/lightglue_matches.h5)
#    -k controls how many nearest neighbors (ENU) each image matches to
# ============================================================
python3 match_lightglue_knn.py \
  --normalized-root "$NORM" \
  --dataset Campbell \
  --mode exhaustive \
  --window 8 \
  --device cuda \
  --overwrite-pairs

# output:
#   $ROOT/matches/Campbell/lightglue_matches.h5


# ============================================================
# 3) RANSAC geometric verification (Fundamental matrix)
#    Removes outlier matches and keeps only epipolar-consistent inliers
# ============================================================
python3 verify_matches_ransac.py \
  --features-h5 "$ROOT/features/Campbell/superpoint_features.h5" \
  --matches-h5  "$ROOT/matches/Campbell/lightglue_matches.h5" \
  --out-h5      "$ROOT/matches/Campbell/verified_matches.h5" \
  --mode fundamental \
  --ransac-thresh 2.5 \
  --min-inliers 20 \
  --overwrite

# output:
#   $ROOT/matches/Campbell/verified_matches.h5


# ============================================================
# 4) Incremental SfM (bootstrap intrinsics from assumed FOV)
#    Produces a sparse 3D point cloud + camera poses
# ============================================================
python3 incremental_sfm.py \
  --features-h5 "$ROOT/features/Campbell/superpoint_features.h5" \
  --verified-matches-h5 "$ROOT/matches/Campbell/verified_matches.h5" \
  --out-dir "$ROOT/sfm/Campbell" \
  --fov-deg 60 \
  --pnp-min-inliers 20 \
  --pnp-reproj-thresh 8.0

# outputs:
#   $ROOT/sfm/Campbell/points.ply
#   $ROOT/sfm/Campbell/points_rgb.ply   (if you're using the colored version)
#   $ROOT/sfm/Campbell/cameras.json


# ============================================================
# View results:
#   - Open points.ply or points_rgb.ply in CloudCompare / MeshLab
#   - cameras.json contains R,t,K per posed camera
# ============================================================
