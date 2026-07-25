# mmaction2-for-badminton
Badminton action recognition using MMAction2 with skeleton-based deep learning models (STGCN++, PoseC3D) for badminton stroke classification.
# MMAction2 for Badminton

基於 **MMAction2** 與 **STGCN++** 的羽球 3D 人體骨架動作辨識專案。

本專案將多視角攝影機取得的 2D 人體姿態資料，透過三角測量轉換為 3D Skeleton，並根據人工標註的動作區間建立 MMAction2 `PoseDataset` 格式資料集，最後使用 STGCN++ 進行羽球動作分類。

## 專案功能

本專案包含以下功能：

* 羽球影片動作區間標註
* 多視角 2D Pose 資料對齊
* 雙相機三角測量
* COCO 17 點 3D Skeleton 產生
* 動作片段切割
* MMAction2 `.pkl` 資料集建立
* STGCN++ 模型訓練與測試
* 自訂 Skeleton Graph Layout
* 支援 COCO、MediaPipe Pose 與 MediaPipe Hands

## 動作類別

目前支援三種羽球動作：

| Label ID | 英文名稱                | 中文名稱   |
| -------: | ------------------- | ------ |
|        0 | `forehand`          | 正手擊球   |
|        1 | `backhand`          | 反手擊球   |
|        2 | `overhead_forehand` | 正手過頭擊球 |

若增加新的動作類別，必須同步修改：

* 標註工具中的 `LABEL_MAP`
* 資料集產生程式中的 `label_map`
* MMAction2 Config 中的 `num_classes`

## 系統流程

```text
多視角羽球影片
        ↓
2D 人體姿態估測
        ↓
Pose CSV 補點與同步
        ↓
雙相機三角測量
        ↓
COCO 17 點 3D Skeleton
        ↓
人工標註動作區間
        ↓
切割 Skeleton 動作片段
        ↓
建立 MMAction2 PoseDataset
        ↓
STGCN++ 模型訓練
        ↓
羽球動作分類
```

## 建議目錄結構

```text
mmaction2-for-badminton/
├── README.md
├── tools/
│   ├── badminton_label_tool.py
│   ├── make_3d_skeleton_from_filled_csv.py
│   ├── make_stgcn_dataset_multi.py
│   └── check_pkl.py
├── data/
│   └── badminton/
│       ├── labels/
│       ├── skeletons/
│       └── badminton_3d_all.pkl
├── configs/
│   └── skeleton/
│       └── stgcnpp/
│           └── stgcnpp_badminton_3d.py
└── mmaction/
    └── models/
        └── utils/
            └── graph.py
```

實際 MMAction2 專案位置可為：

```text
C:\D\NCTU_CS\Thesis\mmaction\mmaction2
```

自訂 Config 放置於：

```text
C:\D\NCTU_CS\Thesis\mmaction\mmaction2\
└── configs\
    └── skeleton\
        └── stgcnpp\
            └── stgcnpp_badminton_3d.py
```

## 環境需求

建議使用獨立的 Conda 環境。

```bash
conda create -n mmaction2 python=3.10 -y
conda activate mmaction2
```

安裝 PyTorch 後，再安裝 OpenMMLab 相關套件：

```bash
pip install -U openmim
mim install mmengine
mim install mmcv
```

從原始碼安裝 MMAction2：

```bash
cd C:\D\NCTU_CS\Thesis\mmaction\mmaction2
pip install -v -e .
```

安裝資料處理套件：

```bash
pip install numpy pandas opencv-python
```

確認 MMAction2 安裝位置：

```bash
python -c "import mmaction; print(mmaction.__file__)"
```

輸出路徑應指向目前專案內的 `mmaction` 資料夾。

## 一、標註羽球動作

使用：

```text
badminton_label_tool.py
```

設定影片路徑：

```python
VIDEO_PATH = r"C:\path\to\CameraReader_0.mp4"
```

執行：

```bash
python badminton_label_tool.py
```

### 操作按鍵

| 按鍵      | 功能                |
| ------- | ----------------- |
| `Space` | 播放或暫停             |
| `s`     | 設定動作起始影格          |
| `e`     | 設定動作結束影格          |
| `1`     | Forehand          |
| `2`     | Backhand          |
| `3`     | Overhead Forehand |
| `a`     | 往前移動 1 frame      |
| `d`     | 往後移動 1 frame      |
| `A`     | 往前移動 4 frames     |
| `D`     | 往後移動 4 frames     |
| `q`     | 結束並儲存             |

輸出 CSV 格式：

```csv
sample_id,start_frame,end_frame,label
s0001,120,150,forehand
s0002,260,295,backhand
s0003,410,448,overhead_forehand
```

## 二、產生 3D Skeleton

使用：

```text
make_3d_skeleton_from_filled_csv.py
```

設定輸入資料夾：

```python
INPUT_PATH = r"C:\path\to\2026-04-09_19-12-21"
```

設定兩個視角的 Pose CSV：

```python
POSE_A_CSV = rf"{INPUT_PATH}\filled_pose_akima\Pose_0_filled.csv"
POSE_B_CSV = rf"{INPUT_PATH}\filled_pose_akima\Pose_2_filled.csv"
```

設定相機編號：

```python
CAM_A = 0
CAM_B = 2
```

程式會讀取：

```text
CameraReader_0.cfg
CameraReader_2.cfg
```

設定檔中必須包含相機 Projection Matrix：

```ini
[Other]
projection_mat = [...]
```

若 Pose 座標為像素座標：

```python
COORD_IS_NORMALIZED = False
```

若 Pose 座標已正規化到 `0～1`：

```python
COORD_IS_NORMALIZED = True
NORM_W = 640
NORM_H = 480
```

執行：

```bash
python make_3d_skeleton_from_filled_csv.py
```

輸出檔案：

```text
badminton_3d_skeleton_<video_name>_filled.npy
badminton_3d_frame_ids_<video_name>_filled.npy
trajectory_<video_name>_filled.csv
```

Skeleton Array Shape：

```text
(T, 17, 3)
```

其中：

| 維度   | 說明           |
| ---- | ------------ |
| `T`  | Skeleton 影格數 |
| `17` | COCO 人體關節數   |
| `3`  | x、y、z 三維座標   |

## 三、建立 MMAction2 資料集

使用：

```text
make_stgcn_dataset_multi.py
```

設定影片清單：

```python
VIDEO_LIST = [
    {
        "name": "2026-04-09_19-13-28",
        "skeleton":
            "badminton_3d_skeleton_2026-04-09_19-13-28_filled.npy",
        "frames":
            "badminton_3d_frame_ids_2026-04-09_19-13-28_filled.npy",
        "labels":
            "labels_20260523_012836.csv",
    },
    {
        "name": "2026-04-09_19-12-21",
        "skeleton":
            "badminton_3d_skeleton_2026-04-09_19-12-21_filled.npy",
        "frames":
            "badminton_3d_frame_ids_2026-04-09_19-12-21_filled.npy",
        "labels":
            "labels_20260523_234410.csv",
    },
]
```

設定類別：

```python
label_map = {
    "forehand": 0,
    "backhand": 1,
    "overhead_forehand": 2,
}
```

執行：

```bash
python make_stgcn_dataset_multi.py
```

輸出：

```text
badminton_3d_all.pkl
```

建議將資料集放置於：

```text
mmaction2/
└── data/
    └── badminton/
        └── badminton_3d_all.pkl
```

## PKL 資料格式

資料集整體格式：

```python
{
    "split": {
        "train": [...],
        "val": [...],
        "test": [...]
    },
    "annotations": [...]
}
```

每個動作樣本格式：

```python
{
    "frame_dir": "2026-04-09_19-13-28_s0001",
    "total_frames": 25,
    "label": 0,
    "keypoint": keypoint_array
}
```

Keypoint Shape：

```text
(M, T, V, C)
```

本專案使用：

```text
M = 1
T = 動作片段影格數
V = 17
C = 3
```

例如：

```text
(1, 25, 17, 3)
```

## 四、STGCN++ Config

本專案使用：

```text
configs/skeleton/stgcnpp/stgcnpp_badminton_3d.py
```

Base Config：

```python
_base_ = (
    'stgcnpp_8xb16-joint-u100-80e_'
    'ntu60-xsub-keypoint-3d.py'
)
```

資料集設定：

```python
dataset_type = 'PoseDataset'
ann_file = 'data/badminton/badminton_3d_all.pkl'
```

模型設定：

```python
model = dict(
    backbone=dict(
        graph_cfg=dict(
            layout='coco',
            mode='spatial',
        ),
    ),
    cls_head=dict(
        num_classes=3,
    ),
)
```

主要參數：

| 參數                | 設定          |
| ----------------- | ----------- |
| Model             | STGCN++     |
| Dataset           | PoseDataset |
| Skeleton Layout   | COCO        |
| Number of Joints  | 17          |
| Number of Persons | 1           |
| Number of Classes | 3           |
| Clip Length       | 32          |
| Graph Mode        | Spatial     |

### Training Pipeline

```python
train_pipeline = [
    dict(type='PreNormalize3D'),
    dict(
        type='GenSkeFeat',
        dataset='coco',
        feats=['j'],
    ),
    dict(
        type='UniformSampleFrames',
        clip_len=32,
    ),
    dict(type='PoseDecode'),
    dict(
        type='FormatGCNInput',
        num_person=1,
    ),
    dict(type='PackActionInputs'),
]
```
### Clip Sampling

本專案使用 `UniformSampleFrames` 將每個 Skeleton 動作片段取樣為固定長度，作為 STGCN++ 的輸入。

```python
dict(
    type='UniformSampleFrames',
    clip_len=32,
)
```

#### 參數說明

| 參數 | 說明 |
|------|------|
| `clip_len=32` | 每個樣本輸入固定 **32 個 Skeleton Frames**。若原始片段長度不足，MMAction2 會自動補點或重複取樣；若長度超過 32，則會均勻取樣 (Uniform Sampling)。 |
| `num_clips=1` | 使用 MMAction2 預設值，每個樣本只取 **一個 Clip** 作為模型輸入。 |

#### STGCN++ Input Shape

本專案每筆資料輸入 STGCN++ 的 Tensor Shape 為：

```text
(M, T, V, C)
= (1, 32, 17, 3)
```

其中：

| 維度 | 說明 |
|------|------|
| `M = 1` | Number of Persons |
| `T = 32` | Skeleton Frames (`clip_len`) |
| `V = 17` | COCO 人體關節數 |
| `C = 3` | x、y、z 三維座標 |

目前僅使用 Joint Feature：

```python
feats=['j']
```

其中 `j` 代表 Joint Feature，也就是人體關節的三維座標。

由於本專案使用 COCO 17 點人體骨架，因此 `GenSkeFeat` 應設定為：

```python
dict(
    type='GenSkeFeat',
    dataset='coco',
    feats=['j'],
)
```

`GenSkeFeat` 的 `dataset` 參數用來指定 Skeleton Feature 所採用的關節格式，而 STGCN++ Backbone 中的 `graph_cfg` 則用來指定關節連接關係與 Graph 鄰接矩陣。

因此兩者都應與 COCO 17 點骨架一致：

```python
dict(
    type='GenSkeFeat',
    dataset='coco',
    feats=['j'],
)
```

```python
graph_cfg=dict(
    layout='coco',
    mode='spatial',
)
```

本專案目前只使用 Joint Feature，尚未啟用 Bone Feature。

## 五、Skeleton Graph Layout

Graph 定義檔位於：

```text
mmaction/
└── models/
    └── utils/
        └── graph.py
```

目前支援：

| Layout      | 關節數 | 用途              |
| ----------- | --: | --------------- |
| `openpose`  |  18 | OpenPose 人體骨架   |
| `nturgb+d`  |  25 | NTU RGB+D 人體骨架  |
| `coco`      |  17 | COCO 人體骨架       |
| `hand21`    |  21 | MediaPipe Hands |
| `mediapipe` |  33 | MediaPipe Pose  |

### 本專案新增的 Layout

相較於原始版本，本專案額外加入：

```text
hand21
mediapipe
```

允許的 Layout 清單修改為：

```python
assert layout in [
    'openpose',
    'nturgb+d',
    'coco',
    'hand21',
    'mediapipe',
]
```

### MediaPipe Hands

```python
elif layout == 'hand21':
    self.num_node = 21
    self.center = 0
```

此 Layout 使用 MediaPipe Hands 21 個關節，並以 Wrist 作為 Center Node。

### MediaPipe Pose

```python
elif layout == 'mediapipe':
    self.num_node = 33
    self.center = 0
```

此 Layout使用 MediaPipe Pose 33 個人體關節。

### COCO Layout

本羽球動作辨識模型使用：

```python
graph_cfg=dict(
    layout='coco',
    mode='spatial',
)
```

COCO 17 點 Graph 已經內建於 `graph.py`，不需要額外修改。

## 六、模型訓練

進入 MMAction2 根目錄：

```bat
cd /d C:\D\NCTU_CS\Thesis\mmaction\mmaction2
```

執行：

```bat
python tools/train.py ^
configs/skeleton/stgcnpp/stgcnpp_badminton_3d.py
```

也可以指定輸出位置：

```bat
python tools/train.py ^
configs/skeleton/stgcnpp/stgcnpp_badminton_3d.py ^
--work-dir work_dirs/stgcnpp_badminton_3d_all
```

PowerShell：

```powershell
python tools/train.py `
    configs/skeleton/stgcnpp/stgcnpp_badminton_3d.py `
    --work-dir work_dirs/stgcnpp_badminton_3d_all
```



## 七、Training Logs

MMAction2 會將訓練結果儲存在 `work_dir`。

若未指定 `--work-dir`，預設會建立於：

```text
mmaction2/
└── work_dirs/
    └── stgcnpp_badminton_3d/
```

若指定：

```bash
python tools/train.py ^
configs/skeleton/stgcnpp/stgcnpp_badminton_3d.py ^
--work-dir work_dirs/stgcnpp_badminton_3d_all
```

則所有訓練結果都會儲存在：

```text
work_dirs/
└── stgcnpp_badminton_3d_all/
```

### 訓練輸出內容

```text
work_dirs/
└── stgcnpp_badminton_3d_all/
    ├── 20260726_143215.log
    ├── 20260726_143215.json
    ├── best_acc_top1_epoch_18.pth
    ├── epoch_1.pth
    ├── epoch_2.pth
    ├── ...
    ├── last_checkpoint
    └── vis_data/
        ├── events.out.tfevents.xxxxxxxxx
        └── config.py
```

### 檔案說明

| 檔案 | 說明 |
|------|------|
| `*.log` | 訓練過程文字紀錄。 |
| `*.json` | 每個 Epoch 的 Loss、Accuracy 等訓練資訊。 |
| `best_acc_top1_epoch_xx.pth` | Top-1 Accuracy 最佳模型。 |
| `epoch_xx.pth` | 每個 Epoch 儲存的 Checkpoint。 |
| `last_checkpoint` | 最近一次訓練的 Checkpoint。 |
| `vis_data/` | TensorBoard 所需的 Log 檔案。 |

### 如何確認 Log 儲存位置

訓練開始後，Terminal 通常會顯示：

```text
Logs will be saved to
work_dirs/stgcnpp_badminton_3d_all
```

若忘記 `work_dir`，可查看：

**Windows**

```bat
dir work_dirs
```

**Linux / macOS**

```bash
ls work_dirs
```

通常最新建立的資料夾就是本次訓練的輸出目錄。


## 八、模型測試

```bat
python tools/test.py ^
configs/skeleton/stgcnpp/stgcnpp_badminton_3d.py ^
work_dirs/stgcnpp_badminton_3d_all\best_acc_top1_epoch_XX.pth
```

請將：

```text
best_acc_top1_epoch_XX.pth
```

替換為實際產生的模型檔名。



## 九、Confusion Matrix

完成模型測試後，可使用 MMAction2 提供的工具產生混淆矩陣（Confusion Matrix），分析各動作類別的分類結果。

### Step 1：輸出模型預測結果

首先使用 `tools/test.py` 將模型預測結果輸出為 `result.pkl`：

```bash
python tools/test.py ^
configs/skeleton/stgcnpp/stgcnpp_badminton_3d.py ^
work_dirs/stgcnpp_badminton_3d_all\best_acc_top1_epoch_18.pth ^
--dump result.pkl
```

完成後會產生：

```text
result.pkl
```

### Step 2：產生 Confusion Matrix

```bash
python tools/analysis_tools/confusion_matrix.py ^
configs/skeleton/stgcnpp/stgcnpp_badminton_3d.py ^
result.pkl ^
work_dirs/stgcnpp_badminton_3d_all\confusion_matrix.png
```

完成後將產生：

```text
work_dirs/
└── stgcnpp_badminton_3d_all/
    └── confusion_matrix.png
```

### 如何解讀 Confusion Matrix

Confusion Matrix 的列（Rows）代表 **Ground Truth（真實標籤）**，欄（Columns）代表 **Prediction（模型預測）**。

例如：

| Ground Truth \\ Prediction | Forehand | Backhand | Overhead |
|----------------------------|---------:|---------:|---------:|
| Forehand                   | 48 | 2 | 0 |
| Backhand                   | 3 | 45 | 2 |
| Overhead                   | 0 | 1 | 49 |

說明：

- 主對角線數值越高，表示分類越正確。
- 非對角線代表誤判，例如 Forehand 被分類成 Backhand。
- 可藉此分析哪些動作最容易混淆，作為增加資料或改善模型的依據。


## 十、TensorBoard

啟動 TensorBoard：

```bash
tensorboard --logdir work_dirs/stgcnpp_badminton_3d_all
```

瀏覽器開啟：

```text
http://localhost:6006
```

可以查看：

* Training Loss
* Validation Loss
* Top-1 Accuracy
* Learning Rate
* Training Epoch

## 十一、資料集切分

正式實驗時，`train`、`val` 與 `test` 不可使用相同樣本。

錯誤示例：

```python
"split": {
    "train": split,
    "val": split,
    "test": split,
}
```

這會造成資料洩漏，使驗證與測試準確率異常偏高。

建議比例：

```text
Training   70%
Validation 15%
Testing    15%
```

更建議採用：

* Video-level split
* Subject-level split
* Session-level split

同一支影片或同一位受試者的相似動作，不應同時出現在 Training 與 Testing Dataset。

Config 應分別使用：

```python
split='train'
```

```python
split='val'
```

```python
split='test'
```

## 十二、注意事項

* 多視角攝影機必須完成時間同步。
* 不同相機的 Pose CSV 必須能依 `frame_id` 對齊。
* Projection Matrix 必須與實際相機相符。
* Label 名稱必須與 `label_map` 完全一致。
* 過短的 Skeleton 片段應略過或另外處理。
* 訓練前應檢查 Skeleton 是否包含 `NaN` 或無效座標。
* `num_classes` 必須與實際動作類別數相同。
* `num_person=1` 代表每個片段只保留一位球員。
* 修改 MMAction2 原始碼後，建議使用 Editable Install。

## 十三、未來工作

後續可擴充：

* Smash 殺球
* Clear 高遠球
* Drop 吊球
* Drive 平抽球
* Serve 發球
* Lift 挑球
* Net Shot 網前球
* 自動動作切割
* 動作開始與結束時間偵測
* 多人 Skeleton Action Recognition
* Subject-independent Evaluation
* PoseC3D 模型比較
* RGB 與 Skeleton 多模態融合
* 即時羽球動作辨識
* Player Load 分析
* MET 能量消耗估測
* 手腕與手臂動作特徵分析

## 十四、致謝

本專案基於以下開源專案開發：

* [MMAction2](https://github.com/open-mmlab/mmaction2)
* [MMEngine](https://github.com/open-mmlab/mmengine)
* [MMCV](https://github.com/open-mmlab/mmcv)
* [PyTorch](https://github.com/pytorch/pytorch)

## 十五、授權

本專案主要用於學術研究、羽球動作辨識與運動科學分析。

使用本專案前，請同時遵循 MMAction2 與相關 OpenMMLab 專案的授權條款。
