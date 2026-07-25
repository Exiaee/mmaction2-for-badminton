import numpy as np
import pandas as pd
import pickle

VIDEO_LIST = [

    {
        "name":"2026-04-09_19-13-28",
        "skeleton":"badminton_3d_skeleton_2026-04-09_19-13-28_filled.npy",
        "frames":"badminton_3d_frame_ids_2026-04-09_19-13-28_filled.npy",
        "labels":"labels_20260523_012836.csv"
    },

    {
        "name":"2026-04-09_19-12-21",
        "skeleton":"badminton_3d_skeleton_2026-04-09_19-12-21_filled.npy",
        "frames":"badminton_3d_frame_ids_2026-04-09_19-12-21_filled.npy",
        "labels":"labels_20260523_234410.csv"
    }

]

label_map = {
    "forehand":0,
    "backhand":1,
    "overhead_forehand":2
}

annotations=[]
split=[]

for video in VIDEO_LIST:

    skeleton=np.load(video["skeleton"])
    frame_ids=np.load(video["frames"])
    labels=pd.read_csv(video["labels"])

    for _,row in labels.iterrows():

        start=int(row["start_frame"])
        end=int(row["end_frame"])

        mask=(
            (frame_ids>=start)&
            (frame_ids<=end)
        )

        segment=skeleton[mask]

        if len(segment)<8:
            continue

        sample_id=(
            f'{video["name"]}_'
            f'{row["sample_id"]}'
        )

        annotations.append({

            "frame_dir":sample_id,

            "total_frames":
            len(segment),

            "label":
            label_map[row["label"]],

            "keypoint":
            segment[None].astype(np.float32)

        })

        split.append(sample_id)

data={

    "split":{
        "train":split,
        "val":split,
        "test":split
    },

    "annotations":
    annotations
}

with open(
    "badminton_3d_all.pkl",
    "wb"
) as f:

    pickle.dump(
        data,
        f
    )

print(
    "samples:",
    len(annotations)
)