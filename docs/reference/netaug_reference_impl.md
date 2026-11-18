# 參考實作分析：`netaug/`（YOLOv8 版 NetAug）

> 來源：專案根目錄的 `netaug/` 資料夾。這是作者在 2023 年寫的實作，建在 ultralytics YOLOv8 上。
> **這個資料夾是唯讀參考資料，不可修改。** 本文件只做整理與分析。
> `netaug/` 被列在 `.gitignore` 中，不會推上遠端 repo；文件中提到的檔名只存在於作者的本機。
> 原始開發專案位置（捷徑指向）：`D:\2025_11_12\PycharmProjects\ultralytics_netaug`。

## 1. 一句話總結

把小模型（base）嵌進一個寬度放大的大模型（max）。所有寬度共用同一組權重，小模型取前面的通道。訓練時，每一步同時算 base 寬度與「增強寬度」的 loss。推論與匯出時只保留 base 寬度。

## 2. 檔案地圖

| 檔案 | 角色 |
|---|---|
| `layers.py` | 動態層：`DynamicModule`、`DynamicConv2d`、`DynamicBatchNorm2d`、`DynamicConv`、`DynamicBottleneck`、`DynamicC2f`、`DynamicSPPF`、`DynamicDetect` |
| `task.py` | `NetAugDetectionModel`：把 YOLO yaml 解析成動態模型，提供 `set_active` / `set_base` / `set_aug` / `set_max` / `export_module` / `sort_channels` / `loss_aug` |
| `loss.py` | `KDLoss`（KL 散度 + 交叉熵）與 `v8DetectionLogitKDLoss`（偵測用的 KD loss） |
| `netaug.py` | `NetAugTrainer`（**KD 版**）：每步呼叫 `loss_aug`；含 `save_model`、`reset_batch_norm` |
| `netaug_1.py` | `NetAugTrainer`（**原始版**）：先預訓練 max 模型，再用 base loss + aug loss |
| `utils/torch_utils.py` | `sort_param`、`sort_norm`、`calc_importance`（L1 重要度排序）、`groups_sorted_idx`、`random_sample` 等 |
| `utils/reset_utils.py` | `AverageMeter` |
| `utils/export_utils.py` | `get_parent_class_name`：匯出時把 `Dynamic*` 換回原本的 ultralytics 類別名 |
| `default.yaml` / `default1.yaml` / `default2.yaml` | ultralytics 訓練設定加上 NetAug 參數（見第 6 節） |
| `yolov8n_dynamic.yaml` | YOLOv8 模型設定，多了 `exclude: [DynamicC2f]`，且 n 尺度寬度改成 0.15 |
| `yolov8n_dynamic1.yaml` | 與 `yolov8n_dynamic.yaml` 內容相同 |

## 3. 核心機制

### 3.1 動態層（權重切片）

- 每個動態層用「最大寬度」建立完整權重。
- 層上有 `active_in_channels`、`active_out_channels`（BN 是 `active_num_features`）。
- forward 時只取前段權重：`weight[:active_out, :active_in]`。BN 的 `weight`、`bias`、`running_mean`、`running_var` 也只取前段。
- forward 會檢查輸入通道數是否等於 `active_in_channels`，不一致就丟錯。
- `export_module()` 依目前的 active 寬度建出一般的 `nn.Conv2d` / `nn.BatchNorm2d` / ultralytics 模組，並複製切好的權重。
- `active_state_dict()` 遞迴收集切片後的權重。

### 3.2 寬度的定義

- 解析 yaml 時，每層輸出通道 `c2 = make_divisible(make_divisible(c2 * width, 8) * max_width, 1)`。也就是說，建出來的是 **max 模型**。
- 增強寬度清單：`aug_width = linspace(1.0, max_width, num_points) / max_width`。
  - 例：`max_width=2, num_points=2` → `[0.5, 1.0]`。
  - `aug_width[0]` 是 base（原始寬度），`aug_width[-1]` 是 max。
- `set_base()`：全部層用 `aug_width[0]`。
- `set_max()`：全部層用 `aug_width[-1]`。
- `set_aug()`：**每一層各自**從 `aug_width` 隨機抽一個比例（`random_choices`）。所以一次取樣的增強網路，各層寬度可能不同。
- yaml 中 `exclude` 清單的類別（預設 `DynamicC2f`）在 `set_aug()` 時固定用 base 寬度。
- 偵測頭 `DynamicDetect` 的隱藏寬度由輸入通道推算，所以會跟著前面層的寬度變動。

### 3.3 只做寬度增強

- `max_depth` 有傳入、有存下來，但**沒有任何地方使用**。深度增強未實作。

### 3.4 通道排序（`sort_channels`，選用）

- 依權重的 L1 範數排序輸出通道，把重要的通道排到前面，讓 base 取到的是最重要的通道。
- `DynamicC2f` 依 `cv2` 的輸入權重排序，再用 `groups_sorted_idx` 分組套到 `cv1` 和各個 bottleneck。這只支援 bottleneck `e=1.0`。
- `DynamicSPPF` 依 `cv2` 的輸入權重排 `cv1`。
- `DynamicDetect.sort_channels()` 是空的。
- 訓練時每 50 個 epoch 執行一次（`sort_channel: True` 時）。
- **注意**：排序只重排了該層的輸出通道與 BN（`dim=0`），**沒有**同步重排下游層的輸入維度（`dim=1`）。例如 `DynamicC2f` 用 `cv2` 的輸入權重算重要度，卻沒有重排 `cv2.conv.weight` 的輸入維度。所以排序後網路的運算結果會改變。官方實作會把同一個排列套用到所有相連的層，見 [netaug_paper.md](netaug_paper.md) 第 5.1 節。

### 3.5 重設 BN 統計值（`reset_batch_norm`，選用）

- 理由：訓練時 BN 的 running 統計值混了不同寬度的資料，對 base 模型不準。
- 做法：
  1. 複製 EMA 模型，切到 base 寬度。
  2. 把每個 BN 的 forward 換成「用本批平均值與變異數做正規化，並累計到 `AverageMeter`」。
  3. 關掉資料增強，跑過一次整個訓練集。
  4. 把累計的平均值與變異數寫回 EMA 模型與訓練模型的 BN 前段。
- 在驗證前執行（`reset_bn: True` 時）。

### 3.6 匯出

- `NetAugDetectionModel.export_module()` 先 `set_base()`，再把每個動態層換成一般層，得到標準的 ultralytics `DetectionModel`。
- `save_model()` 存兩份：
  - `last.pt` / `best.pt`：匯出後的 base 模型，可直接用一般 YOLO 推論。
  - `aug_last.pt` / `aug_best.pt`：完整的動態模型（含增強寬度的權重）。

## 4. 兩種訓練演算法

### 4.1 `netaug_1.py`：原始版

每一步：

```text
if epoch < aug_pretrain:
    set_max();  loss = L(max 模型)            # 先預訓練最大模型
else:
    set_base(); loss = L(base 模型)
    if epoch < stop_width_aug:
        set_aug(); loss += L(增強模型)       # 權重 = 1，沒有 alpha 係數
```

### 4.2 `netaug.py`：KD 版（較新，`loss_aug`）

每一步：

```text
set_base(); preds = base 模型輸出
if epoch < stop_width_aug:
    set_aug(); aug_preds = 增強模型輸出
    loss = KD_loss(學生 = base, 老師 = aug) + L(增強模型)
else:
    loss = L(base 模型)
```

`v8DetectionLogitKDLoss` 的內容：

- base 模型用自己的預測做 label assignment，算一般的 box / cls / dfl loss。
- 再用老師（aug 模型）預測做一次 label assignment，得到「老師的目標框與分數」。
- 額外加上：
  - `kd_cls` × KL 散度（base 分數 vs 老師目標分數）。
  - `kd_box`、`kd_dfl` × 用老師目標算的 box / dfl loss。
- 沒有 max 預訓練階段。

## 5. 已知限制與可能的問題

這些是閱讀程式碼時發現的地方。做新套件時要決定是否沿用。

1. **緊耦合舊版 ultralytics**：大量使用 `ultralytics.yolo.*`（8.0.x 時期的 API），新版 ultralytics 已改路徑。
2. **不支援分組卷積**：`DynamicConv2d` 標了 TODO，`groups > 1`（例如 depthwise）時切片會錯。
3. **`max_depth` 沒有作用**：只有寬度增強。
4. **寬度取整**：`int(out_channels * 比例)` 用浮點數相乘後截斷。比例像 1/3 時，可能因浮點誤差少 1 個通道。
5. **KD 的 softmax 維度**：`KDLoss` 在 `(batch, anchors, classes)` 張量上用 `dim=1` 做 softmax，這是 anchor 維度，不是類別維度。可能不是原意，需要再確認。
6. **重複建模**：`NetAugDetectionModel.__init__` 先用父類別建一次一般模型，再重建一次動態模型。
7. **設定值互相矛盾**：`default.yaml` 的 `epochs: 100`，但 `aug_pretrain: 300`、`stop_width_aug: 600`。用 `netaug_1.py` 時整個訓練都只在預訓練 max 模型，實際使用時應該有用命令列覆寫。
8. **重複呼叫**：`netaug.py` 在呼叫 `loss_aug` 前先 `set_base()`，`loss_aug` 內又呼叫一次。
9. **EMA 與動態寬度**：EMA 模型也是動態模型，驗證前要記得 `set_base()`（程式有做）。
10. **`__pycache__` 是 cpython-38**：當時用 Python 3.8。
11. **通道排序會改變網路的運算結果**：只重排輸出通道，沒重排下游層的輸入維度（見第 3.4 節）。這可能是 `sort_channel` 預設關閉的原因。
12. **授權**：`netaug.py` 開頭帶有 `Ultralytics YOLO 🚀, GPL-3.0 license` 字樣。新套件若沿用這部分程式碼，要考慮 GPL-3.0 的影響。

## 6. 設定參數（`default*.yaml` 尾段）

| 參數 | `default.yaml` | `default1.yaml` | `default2.yaml` | 說明 |
|---|---|---|---|---|
| `max_width` | 2.0 | 3.0 | 2.0 | max 模型相對 base 的寬度倍數 |
| `max_depth` | 1.0 | 1.0 | 1.0 | 未使用 |
| `num_points` | 2 | 3 | 5 | 寬度清單的點數 |
| `aug_pretrain` | 300 | — | — | max 預訓練的 epoch 數（`netaug_1.py` 用） |
| `stop_width_aug` | 600 | 600 | 600 | 這個 epoch 之後停止寬度增強 |
| `sort_channel` | False | False | False | 是否每 50 epoch 排序通道 |
| `reset_bn` | False | False | False | 是否在驗證前重設 BN |
| `use_kd` | True | — | — | KD 開關（程式中沒有讀取，KD 版直接使用） |
| `kd_box` / `kd_cls` / `kd_dfl` | 0.75 / 0.05 / 0.15 | — | — | KD 各項權重 |
| `temperature` / `alpha` / `beta` | 2 / 0.5 / 0.5 | — | — | `KDLoss` 參數 |

`default.yaml` 對應 `netaug.py`（KD 版）。`default1.yaml`、`default2.yaml` 沒有 KD 參數，對應較早的實驗。

## 7. 對新套件可重用的設計

- **權重切片的動態層**：做法通用，可以脫離 ultralytics，做成純 PyTorch 的 `DynamicConv2d`、`DynamicBatchNorm2d`、`DynamicLinear`。
- **`set_active` 介面**：base / max / 隨機增強三種模式。
- **匯出成一般 `nn.Module`**：讓部署端不需要依賴本套件。
- **選用技巧**：通道排序、BN 重設、max 預訓練、停止增強的時間點、KD loss。

需要通用化的部分：

- 通道之間的相依關係（殘差相加、concat、分組卷積）目前是每個模組手寫。通用套件需要一套方法處理這些相依關係。
- 訓練迴圈目前寫死在 ultralytics trainer 裡。通用套件應該提供可嵌入任意訓練迴圈的 loss 包裝或 hook。
