# MMAction2 ST-GCN++ Code 運作原理整理

> 目標：從 MMAction2 原始碼角度理解 ST-GCN++ 如何把 Skeleton Sequence
> 轉換成動作分類特徵。
> 主要原始碼：`mmaction/models/backbones/stgcn.py`

## 1. 整體概念

ST-GCN++ 的輸入不是 RGB 影像，而是一段人體骨架序列。模型同時學習：

-   **Spatial（空間）**：不同人體關節之間的關係。
-   **Temporal（時間）**：關節／骨架特徵隨 Frame 的變化。
-   **Graph（圖）**：利用人體骨架拓樸定義 Joint 之間的連接。

整體可簡化為：

``` text
Skeleton Sequence
[N, M, T, V, C]
       │
       ▼
Graph / Adjacency Matrix A
       │
       ▼
Data Batch Normalization
       │
       ▼
[N*M, C, T, V]
       │
       ▼
STGCNBlock × 多層
 ├─ unit_gcn       → Spatial feature
 ├─ mstcn          → Temporal feature
 └─ Residual+ReLU
       │
       ▼
High-level Skeleton Feature
       │
       ▼
Classification Head
       │
       ▼
Action Class
```

------------------------------------------------------------------------

## 2. 輸入 Tensor：`N, M, T, V, C`

`STGCN.forward()` 一開始：

``` python
N, M, T, V, C = x.size()
```

各維度意義：

  符號   意義                    以 COCO 17 點 3D Skeleton 為例
  ------ ----------------------- --------------------------------
  `N`    Batch size              例如 32
  `M`    人數                    單人可為 1
  `T`    Frames                  例如 60、100、150
  `V`    Joint 數                COCO = 17
  `C`    每個 Joint 的 feature   3D = `(x,y,z)`

例如：

``` text
x.shape = [32, 1, 60, 17, 3]
```

表示一個 Batch 有 32 段 Skeleton sequence，每段 60 Frames、1 個人、17
個關節，每個關節具有 `(x,y,z)`。

------------------------------------------------------------------------

## 3. 人體 Skeleton 先建立成 Graph

初始化 Backbone 時會建立：

``` python
self.graph = Graph(**graph_cfg)
A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
```

其中 `A` 是 **Adjacency Matrix（鄰接矩陣）**。

人體可視為：

``` text
Shoulder ─ Elbow ─ Wrist
    │
   Hip
    │
   Knee
    │
  Ankle
```

-   Joint = Graph Node
-   Bone connection = Graph Edge

概念上：

```math
A_{ij} =
\begin{cases}
1, & \text{Joint}_i \text{ 與 Joint}_j \text{ 有連接} \\
0, & \text{其他}
\end{cases}
```

GCN 因此知道哪些 Joint 之間需要進行 feature aggregation。

------------------------------------------------------------------------

## 4. Data Batch Normalization 與 Tensor 排列

原始輸入：

``` text
[N, M, T, V, C]
```

程式先重新排列：

``` python
x = x.permute(0, 1, 3, 4, 2).contiguous()
```

接著進行 Batch Normalization，最後整理成：

``` text
[N*M, C, T, V]
```

因此 Backbone 內部主要處理：

``` text
Batch × Channel × Time × Joint
```

對 3D COCO Skeleton：

``` text
[N*M, 3, T, 17]
```

這個排列非常重要：

-   GCN 主要沿著 `V`（Joint）處理空間關係。
-   TCN 主要沿著 `T`（Time）處理時間變化。

------------------------------------------------------------------------

# 5. `STGCNBlock`：整個模型最核心的基本單元

MMAction2 中：

``` python
class STGCNBlock(BaseModule):
```

它本身的核心其實很簡單：

``` python
def forward(self, x):
    res = self.residual(x)
    x = self.tcn(self.gcn(x)) + res
    return self.relu(x)
```

因此數學概念就是：

```math
Y = \mathrm{ReLU}\left(TCN(GCN(X)) + Residual(X)\right)
```

流程：

``` text
                  ┌──── Residual ─────┐
                  │                   │
X ──→ GCN ──→ TCN ─────────────────→ (+) ──→ ReLU ──→ Y
```

所以一個 `STGCNBlock` 做三件主要事情：

1.  GCN：學 Joint 與 Joint 的空間關係。
2.  TCN：學 Skeleton feature 的時間變化。
3.  Residual：保留原始 feature，幫助深層網路訓練。

------------------------------------------------------------------------

# 6. `unit_gcn`：Spatial Graph Convolution

初始化：

``` python
self.gcn = unit_gcn(
    in_channels,
    out_channels,
    A,
    **gcn_kwargs
)
```

它處理的是 **Joint ↔ Joint** 的資訊交換。

例如：

``` text
Right Shoulder ─ Right Elbow ─ Right Wrist
```

傳統 CNN 是在規則 Grid 上做 convolution；GCN 則是在 Skeleton Graph
上依照 `A` 聚合鄰近 Node 的資訊。

可用簡化公式理解：

```math
X_s = GCN(X, A)
```

或概念式：

```math
X' \approx A X W
```

其中：

- \(X\)：Skeleton features
- \(A\)：Graph adjacency matrix
- \(W\)：可學習權重矩陣

所以 GCN 的工作可以理解成：

> **分析某一時間範圍內，各人體關節之間的空間結構與關聯。**

例如羽球揮拍時，Shoulder、Elbow、Wrist 的相對關係會形成重要特徵。

------------------------------------------------------------------------

# 7. ST-GCN++ 的 Adaptive GCN

ST-GCN++ 的重要設定之一是讓 Graph 不完全只依賴人工固定的 Skeleton
topology。

例如設定會透過 `gcn_*` 參數傳入 `unit_gcn`，使 adjacency 可採 adaptive
形式。

傳統固定 Graph：

``` text
Shoulder ─ Elbow ─ Wrist
```

但某些動作中，即使兩個 Joint 沒有直接 Bone
connection，它們之間仍可能具有重要相關性。

例如羽球：

``` text
Right Wrist ↔ Right Shoulder
Right Wrist ↔ Torso
Upper Body ↔ Lower Body
```

因此可以把概念理解為：

```math
A_{\mathrm{fixed}} \rightarrow A_{\mathrm{adaptive}}
```

模型可以在訓練過程中調整 Graph 關係，使其更符合動作分類需求。

------------------------------------------------------------------------

# 8. `TCN`：Temporal Convolution

GCN 完成後：

``` python
x = self.gcn(x)
```

接著：

``` python
x = self.tcn(x)
```

TCN 處理的是：

``` text
Frame t-2 → t-1 → t → t+1 → t+2
```

也就是 Skeleton feature 隨時間如何改變。

例如 Wrist：

``` text
Frame 1   Frame 2   Frame 3   Frame 4
   ●   →     ●   →     ●   →     ●
```

GCN 可以知道「手腕和手肘現在的關係」，而 TCN 則進一步知道：

> **這個關係如何隨時間形成一個動作。**

因此：

```math
X_t = TCN(X_s)
```

------------------------------------------------------------------------

# 9. ST-GCN 與 ST-GCN++：`unit_tcn` vs `mstcn`

`STGCNBlock` 會依設定選擇：

``` python
if tcn_type == 'unit_tcn':
    self.tcn = unit_tcn(...)
elif tcn_type == 'mstcn':
    self.tcn = mstcn(...)
```

普通 ST-GCN 可使用 `unit_tcn`。

ST-GCN++ 的重要設計則是使用：

``` text
mstcn
```

即 **Multi-Scale Temporal Convolution Network**。

概念：

``` text
                     ┌─ Temporal Branch 1 ─┐
                     │                     │
GCN Feature ─────────┼─ Temporal Branch 2 ─┼─→ Fusion
                     │                     │
                     ├─ Temporal Branch 3 ─┤
                     │                     │
                     └─ Other Branch ──────┘
```

目的：

> 使用不同 Temporal receptive fields 同時捕捉短期與較長期的動作變化。

以羽球 Smash 為例：

``` text
準備 → 抬手 → 加速揮拍 → 擊球 → Follow-through
```

短時間尺度可以抓快速揮拍；較長時間尺度則可以描述完整動作過程。

------------------------------------------------------------------------

# 10. Residual Connection

`STGCNBlock`：

``` python
res = self.residual(x)
x = self.tcn(self.gcn(x)) + res
```

如果 input/output channel 相同且 stride = 1：

``` python
self.residual = lambda x: x
```

因此：

```math
Y = F(X) + X
```

其中：

```math
F(X) = TCN(GCN(X))
```

如果 Channel 或時間尺寸不同，則使用 `1×1` temporal convolution：

``` python
self.residual = unit_tcn(
    in_channels,
    out_channels,
    kernel_size=1,
    stride=stride
)
```

Residual 的作用包括：

-   保留原始 feature。
-   改善梯度傳遞。
-   讓多層 STGCNBlock 更容易訓練。

------------------------------------------------------------------------

# 11. ReLU

最後：

``` python
return self.relu(x)
```

因此完整 Block：

```math
\boxed{
Y = \mathrm{ReLU}\left(TCN(GCN(X)) + Residual(X)\right)
}
```

這是理解 `STGCNBlock` 最重要的公式。

------------------------------------------------------------------------

# 12. ST-GCN++ 不是只有一個 `STGCNBlock`

Backbone 會建立多個：

``` python
modules.append(
    STGCNBlock(
        in_channels,
        out_channels,
        A.clone(),
        stride,
        ...
    )
)
```

Forward 時：

``` python
for i in range(self.num_stages):
    x = self.gcn[i](x)
```

注意這裡：

``` python
self.gcn
```

雖然變數名稱叫 `gcn`，實際上它是一個：

``` text
ModuleList[STGCNBlock]
```

所以並不是只呼叫單純 `unit_gcn`。

真正流程是：

``` text
Input
  ↓
STGCNBlock 1
  ↓
STGCNBlock 2
  ↓
STGCNBlock 3
  ↓
...
  ↓
STGCNBlock N
  ↓
Output Feature
```

------------------------------------------------------------------------

# 13. Channel 為什麼從 64 → 128 → 256？

Backbone 有：

``` python
base_channels = 64
ch_ratio = 2
inflate_stages = [5, 8]
```

到指定 Stage 時增加 feature channels。

概念：

``` text
Input XYZ
 C = 3
   ↓
64-dimensional feature
   ↓
64
   ↓
...
128
   ↓
...
256
```

這裡的：

``` text
64 / 128 / 256
```

不是 Joint 數，而是 Network 學習出的 feature dimension。

前層可能偏向：

``` text
Wrist position
Elbow relation
Shoulder relation
Knee movement
```

較深層則逐步組合成更高階的動作 pattern。

------------------------------------------------------------------------

# 14. Temporal Downsampling

Backbone 具有：

``` python
down_stages = [5, 8]
```

對應 stage：

``` python
stride = 1 + (i in down_stages)
```

如果 stage 位於 `down_stages`：

``` text
stride = 2
```

Temporal dimension 因此下降。

例如：

``` text
150 Frames
   ↓
75 Frames
   ↓
38 Frames
```

同時 feature channel 增加：

``` text
150 × 64
    ↓
75 × 128
    ↓
38 × 256
```

概念與 CNN 的 spatial downsampling 類似，只是這裡主要壓縮的是 **Time
dimension**。

------------------------------------------------------------------------

# 15. Backbone 最後輸出什麼？

經過所有 `STGCNBlock`：

``` python
for i in range(self.num_stages):
    x = self.gcn[i](x)
```

最後得到高階 Skeleton feature，例如：

``` text
[N, M, 256, T', V]
```

這不是最終動作類別，而是 **Backbone Feature**。

之後還要交給 Classification Head。

概念：

``` text
ST-GCN++ Backbone
        ↓
Skeleton Feature
        ↓
Global Pooling
        ↓
Classification Head / FC
        ↓
Class Scores
        ↓
Softmax / argmax
        ↓
Action Label
```

------------------------------------------------------------------------

# 16. 用你的羽球資料理解

假設你的資料：

``` text
COCO 17 joints
3D coordinates
60 frames
1 player
5 action classes
```

輸入：

``` text
[N, 1, 60, 17, 3]
```

完整流程：

``` text
3D Skeleton Sequence
[N,1,60,17,3]
        │
        ▼
Graph
COCO 17-joint topology
        │
        ▼
Batch Normalization
        │
        ▼
[N,3,60,17]
        │
        ▼
STGCNBlock
 ├─ unit_gcn
 │    └─ Joint spatial relationships
 │
 ├─ mstcn
 │    └─ Multi-scale temporal motion
 │
 └─ Residual + ReLU
        │
        ▼
STGCNBlock
        │
        ▼
...
        │
        ▼
High-level Skeleton Feature
        │
        ▼
Classification Head
        │
        ▼
5-class scores
        │
        ▼
Predicted Badminton Action
```

------------------------------------------------------------------------

# 17. `STGCNBlock` 一句話理解

``` python
x = self.tcn(self.gcn(x)) + res
```

可以直接翻譯成：

> **先分析人體各關節之間的空間關係，再分析這些骨架特徵隨時間的變化，最後加回原始特徵。**

也就是：

``` text
GCN
= 人的姿態「長什麼樣」

TCN / MSTCN
= 這個姿態「怎麼動」

STGCNBlock
= 同時學「姿態 + 動態」

多層 STGCNBlock
= 從低階 Joint movement 學成高階 Action pattern
```

------------------------------------------------------------------------

# 18. ST-GCN 與 ST-GCN++ 在 Code 層級的重點

不要把它理解成兩份完全不同的 Backbone code。

MMAction2 使用通用的：

``` text
STGCN
 └─ STGCNBlock
      ├─ unit_gcn
      ├─ unit_tcn / mstcn
      └─ residual
```

透過設定改變 Block 內部的行為。

簡化理解：

  項目                ST-GCN              ST-GCN++
  ------------------- ------------------- ------------------------------
  Backbone 基本結構   STGCN               STGCN
  基本單元            STGCNBlock          STGCNBlock
  Spatial             GCN                 改良／Adaptive GCN 設定
  Temporal            unit_tcn            mstcn
  Residual            有                  有
  目的                Skeleton 時空建模   更強的 Skeleton GCN baseline

所以 ST-GCN++ 可以理解為：

> **保留 ST-GCN 的「GCN + TCN + Residual」骨架，但採用更好的 Graph 與
> Multi-Scale Temporal 設計及訓練設定。**

------------------------------------------------------------------------

# 19. Code 呼叫關係

讀 MMAction2 source 時，建議按照這個順序：

``` text
Config
  │
  ▼
RecognizerGCN
  │
  ├─ backbone = STGCN
  │
  └─ cls_head
        │
        ▼
STGCN.__init__()
  │
  ├─ Graph()
  ├─ BatchNorm
  └─ 建立多個 STGCNBlock
        │
        ▼
STGCN.forward()
  │
  ├─ reshape / permute
  ├─ data_bn
  └─ STGCNBlock × N
        │
        ▼
STGCNBlock.forward()
  │
  ├─ residual(x)
  ├─ unit_gcn(x)
  ├─ mstcn(x)
  ├─ + residual
  └─ ReLU
        │
        ▼
Skeleton Feature
        │
        ▼
Classification Head
        │
        ▼
Action Prediction
```

------------------------------------------------------------------------

# 20. 最重要的三層理解

如果只想記住 ST-GCN++ code，可以分成三層。

## Layer 1：Model

``` text
STGCN
```

負責：

-   建 Graph。
-   整理輸入 Tensor。
-   建立多個 STGCNBlock。
-   執行 Backbone。

## Layer 2：Block

``` text
STGCNBlock
```

核心：

``` python
x = self.tcn(self.gcn(x)) + res
```

即：

```math
Spatial \rightarrow Temporal \rightarrow Residual
```

## Layer 3：Operator

``` text
unit_gcn
mstcn
unit_tcn
```

其中：

``` text
unit_gcn
    ↓
Joint-to-Joint spatial relationship

mstcn
    ↓
Multi-scale temporal relationship

Residual
    ↓
Original feature preservation
```

------------------------------------------------------------------------

# 21. 最後總結

MMAction2 中 ST-GCN++ 的 Code 運作可以濃縮成：

``` text
3D Skeleton
     ↓
[N,M,T,V,C]
     ↓
Skeleton Graph A
     ↓
Batch Normalization
     ↓
[N*M,C,T,V]
     ↓
┌────────────────────────┐
│      STGCNBlock        │
│                        │
│  unit_gcn              │
│     ↓                  │
│  Spatial Feature       │
│     ↓                  │
│  mstcn                 │
│     ↓                  │
│  Temporal Feature      │
│     ↓                  │
│  + Residual            │
│     ↓                  │
│  ReLU                  │
└────────────────────────┘
     ↓
 × 多個 Stage
     ↓
High-level Skeleton Feature
     ↓
Classification Head
     ↓
Action Classification
```

核心公式：

```math
\boxed{
Y = \mathrm{ReLU}\left(MSTCN(GCN(X, A)) + Residual(X)\right)
}
```

對羽球動作辨識而言：

-   `Graph`：定義人體骨架。
-   `unit_gcn`：學習 Shoulder、Elbow、Wrist、Hip、Knee
    等關節之間的空間關係。
-   `mstcn`：學習揮拍、跑動、跳躍等動作隨時間的變化。
-   多層 `STGCNBlock`：將低階 Joint movement 組合成高階動作特徵。
-   Classification Head：將 Skeleton feature 分成你的 5 類羽球動作。

------------------------------------------------------------------------

## 參考原始碼

-   MMAction2 `mmaction/models/backbones/stgcn.py`
-   MMAction2 `configs/skeleton/stgcnpp/README.md`
-   PYSKL / ST-GCN++: *PYSKL: Towards Good Practices for Skeleton Action
    Recognition*
