import cv2
import csv
import datetime
from datetime import datetime
date = datetime.now().strftime("%Y%m%d_%H%M%S")
#VIDEO_PATH = r"C:\D\NCTU_CS\Thesis\Lab_Data\dataset\dataset\2026-04-09_19-13-28\CameraReader_0.mp4"
VIDEO_PATH = r"C:\D\NCTU_CS\Thesis\Lab_Data\dataset\dataset\2026-04-09_19-12-21\CameraReader_0.mp4"
OUT_CSV = f"labels_{date}.csv"

LABEL_MAP = {
    ord("1"): "forehand",           # 正手
    ord("2"): "backhand",           # 反手
    ord("3"): "overhead_forehand",  # 正手過頭
}

cap = cv2.VideoCapture(VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

labels = []
current_start = None
frame_id = 0
paused = True

print("Hotkeys:")
print("s = set start frame")
print("e = set end frame, then press 1/2/3")
print("1 = forehand")
print("2 = backhand")
print("3 = overhead_forehand")
print("a = previous frame")
print("d = next frame")
print("A = previous 4 frames")
print("D = next 4 frames")
print("space = play/pause")
print("q = quit and save")

while True:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()
    if not ret:
        break

    show = frame.copy()
    skeleton_id = frame_id // 4

    cv2.putText(show, f"Frame: {frame_id} / {total_frames-1}", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.putText(show, f"SkeletonID: {skeleton_id}", (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(show, "1:Forehand  2:Backhand  3:Overhead Forehand", (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    if current_start is not None:
        cv2.putText(show, f"Start: {current_start}", (30, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Label Tool", show)
    key = cv2.waitKey(30) & 0xFF

    if key == ord("q"):
        break

    elif key == ord(" "):
        paused = not paused

    elif key == ord("s"):
        current_start = frame_id
        print(f"Start frame = {current_start}")

    elif key == ord("e"):
        if current_start is None:
            print("Please press s first.")
            continue

        end_frame = frame_id
        print("Select label: 1=forehand, 2=backhand, 3=overhead_forehand")

        label = None
        while label is None:
            label_key = cv2.waitKey(0) & 0xFF
            if label_key in LABEL_MAP:
                label = LABEL_MAP[label_key]
            elif label_key == ord("q"):
                break

        if label is None:
            continue

        sample_id = f"s{len(labels)+1:04d}"
        labels.append({
            "sample_id": sample_id,
            "start_frame": current_start,
            "end_frame": end_frame,
            "label": label
        })

        print(f"Added: {sample_id}, {current_start}-{end_frame}, {label}")
        current_start = None

    elif key == ord("a"):
        frame_id = max(frame_id - 1, 0)

    elif key == ord("d"):
        frame_id = min(frame_id + 1, total_frames - 1)

    elif key == ord("A"):
        frame_id = max(frame_id - 4, 0)

    elif key == ord("D"):
        frame_id = min(frame_id + 4, total_frames - 1)

    if not paused:
        frame_id = min(frame_id + 1, total_frames - 1)

cap.release()
cv2.destroyAllWindows()

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["sample_id", "start_frame", "end_frame", "label"]
    )
    writer.writeheader()
    writer.writerows(labels)

print(f"Saved {OUT_CSV}")