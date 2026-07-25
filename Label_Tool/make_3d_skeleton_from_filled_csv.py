import cv2
import numpy as np
import pandas as pd
import configparser
import json
import os
from pathlib import Path


# ============================================================
# User settings
# ============================================================
#INPUT_PATH = r"C:\D\NCTU_CS\Thesis\Lab_Data\dataset\dataset\2026-04-09_19-13-28"
INPUT_PATH = r"C:\D\NCTU_CS\Thesis\Lab_Data\dataset\dataset\2026-04-09_19-12-21"

#POSE_A_CSV = rf"{INPUT_PATH}\filled_pose_akima\Pose_0_filled_20260523_015347.csv"
#POSE_B_CSV = rf"{INPUT_PATH}\filled_pose_akima\Pose_2_filled_20260523_015347.csv"

POSE_A_CSV = rf"{INPUT_PATH}\filled_pose_akima\Pose_0_filled_20260520_004440.csv"
POSE_B_CSV = rf"{INPUT_PATH}\filled_pose_akima\Pose_2_filled_20260520_004440.csv"

CAM_A = 0
CAM_B = 2

NUM_KEYPOINTS = 17
FRAME_STEP = 1

# If your CSV keypoints are normalized 0~1, set this True.
# If your CSV keypoints are already pixel coordinates, keep False.
COORD_IS_NORMALIZED = False
NORM_W = 640
NORM_H = 640

# Moving average smoothing window.
SMOOTH_WINDOW = 3

folder_name = Path(INPUT_PATH).name

OUT_SKEL = f"badminton_3d_skeleton_{folder_name}_filled.npy"
OUT_FRAMES = f"badminton_3d_frame_ids_{folder_name}_filled.npy"
OUT_TRAJ = f"trajectory_{folder_name}_filled.csv"


# ============================================================
# COCO 17 keypoint index
# ============================================================
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16


def read_projection_matrix(cfg_path: str) -> np.ndarray:
    parser = configparser.ConfigParser()
    parser.read(cfg_path)

    if "Other" not in parser or "projection_mat" not in parser["Other"]:
        raise KeyError(f"projection_mat not found in {cfg_path}")

    return np.array(
        json.loads(parser["Other"]["projection_mat"]),
        dtype=np.float32
    )


def build_points_2d(row: pd.Series, suffix: str) -> np.ndarray:
    """Return shape (17, 2)."""
    pts = []
    for j in range(NUM_KEYPOINTS):
        x = row[f"kpts_{j}_x_{suffix}"]
        y = row[f"kpts_{j}_y_{suffix}"]

        if COORD_IS_NORMALIZED:
            x *= NORM_W
            y *= NORM_H

        pts.append([x, y])

    return np.array(pts, dtype=np.float32)


def triangulate_skeleton(
    points_2d_A: np.ndarray,
    points_2d_B: np.ndarray,
    projMtx_A: np.ndarray,
    projMtx_B: np.ndarray
) -> np.ndarray:
    """
    points_2d_A/B: (17, 2)
    return: (17, 3)
    """
    points_3d_h = cv2.triangulatePoints(
        projMtx_A,
        projMtx_B,
        points_2d_A.T,
        points_2d_B.T
    )

    # homogeneous normalize
    w = points_3d_h[3]
    w[np.abs(w) < 1e-8] = np.nan

    points_3d_h = points_3d_h / w
    points_3d = points_3d_h[:3].T

    return points_3d.astype(np.float32)


def smooth_with_buffer(buffer: list, new_points: np.ndarray, window: int) -> np.ndarray:
    buffer.append(new_points.copy())

    if len(buffer) > window:
        buffer.pop(0)

    return np.nanmean(np.stack(buffer, axis=0), axis=0).astype(np.float32)


def estimate_player_position(skeleton_3d: np.ndarray) -> np.ndarray:
    """
    Use hip center as player trajectory position.
    This is usually more stable than ankle center.
    """
    left_hip = skeleton_3d[LEFT_HIP]
    right_hip = skeleton_3d[RIGHT_HIP]

    return np.nanmean(
        np.vstack([left_hip, right_hip]),
        axis=0
    )


def main():
    cfg_A = os.path.join(INPUT_PATH, f"CameraReader_{CAM_A}.cfg")
    cfg_B = os.path.join(INPUT_PATH, f"CameraReader_{CAM_B}.cfg")

    projMtx_A = read_projection_matrix(cfg_A)
    projMtx_B = read_projection_matrix(cfg_B)

    pose_A = pd.read_csv(POSE_A_CSV)
    pose_B = pd.read_csv(POSE_B_CSV)

    required_cols = ["frame_id"]
    for j in range(NUM_KEYPOINTS):
        required_cols += [f"kpts_{j}_x", f"kpts_{j}_y"]

    for col in required_cols:
        if col not in pose_A.columns:
            raise KeyError(f"{col} missing in {POSE_A_CSV}")
        if col not in pose_B.columns:
            raise KeyError(f"{col} missing in {POSE_B_CSV}")

    # Align camera A/B by frame_id
    df = pose_A.merge(
        pose_B,
        on="frame_id",
        suffixes=("_A", "_B")
    )

    df = df.sort_values("frame_id").reset_index(drop=True)

    print("Loaded Pose A:", pose_A.shape)
    print("Loaded Pose B:", pose_B.shape)
    print("Aligned frames:", df.shape)

    all_3d_frames = []
    all_frame_ids = []
    trajectory_rows = []

    smooth_buffer = []

    for _, row in df.iterrows():
        frame_id = int(row["frame_id"])

        if FRAME_STEP > 1 and frame_id % FRAME_STEP != 0:
            continue

        points_2d_A = build_points_2d(row, "A")
        points_2d_B = build_points_2d(row, "B")

        skeleton_3d = triangulate_skeleton(
            points_2d_A,
            points_2d_B,
            projMtx_A,
            projMtx_B
        )

        if SMOOTH_WINDOW > 1:
            skeleton_3d = smooth_with_buffer(
                smooth_buffer,
                skeleton_3d,
                SMOOTH_WINDOW
            )

        all_3d_frames.append(skeleton_3d)
        all_frame_ids.append(frame_id)

        pos = estimate_player_position(skeleton_3d)
        trajectory_rows.append({
            "frame_id": frame_id,
            "x": pos[0],
            "y": pos[1],
            "z": pos[2],
        })

    all_3d_frames = np.array(all_3d_frames, dtype=np.float32)
    all_frame_ids = np.array(all_frame_ids, dtype=np.int32)

    np.save(OUT_SKEL, all_3d_frames)
    np.save(OUT_FRAMES, all_frame_ids)

    pd.DataFrame(trajectory_rows).to_csv(
        OUT_TRAJ,
        index=False,
        encoding="utf-8-sig"
    )

    print("Saved:", OUT_SKEL, all_3d_frames.shape)
    print("Saved:", OUT_FRAMES, all_frame_ids.shape)
    print("Saved:", OUT_TRAJ)

    if np.isnan(all_3d_frames).any():
        print("[WARN] NaN exists in 3D skeleton.")
    else:
        print("[OK] No NaN in 3D skeleton.")


if __name__ == "__main__":
    main()
