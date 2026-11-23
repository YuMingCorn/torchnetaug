# 術語對照 / Glossary

- 中文與英文文件使用同一組術語。 / Chinese and English documents use the same terms.
- 程式識別字（如 `NetAugModel`）不翻譯。 / Code identifiers (such as `NetAugModel`) are never translated.
- 每個概念只用一個名稱；「避免」欄列出不要再用的同義詞。 / Each concept has one name; the "Avoid" column lists synonyms not to use.

## 模型 / Models

| 中文 | English | 定義 / Definition | 避免 / Avoid |
|---|---|---|---|
| 網路增強 | Network Augmentation (NetAug) | 訓練時把 base 模型放進共用權重的較大網路一起訓練 / train the base model inside larger networks that share its weights | |
| base 模型 | base model | 使用者提供、最後要部署的原始模型 / the user's original model, which is deployed | 小模型、tiny 模型 / tiny model |
| 增強網路 | augmented network | 包含 base 模型、每步隨機抽樣的較寬或較深網路 / a wider or deeper network that contains the base model, sampled each step | 增強模型 |
| 最大寬度網路 | max-width network | 每個可增強群組都取最大寬度的網路 / the network with every free group at its maximum width | 最大寬度模型、動態網路 |
| 動態模型 | dynamic model | 以最大寬度存權重、可切換到任何寬度與深度的模型 / the model that stores weights at maximum width and can switch to any width and depth | 動態網路 |
| 動態層 | dynamic layer | 以最大寬度存權重、執行時切片的層 / a layer stored at maximum width and sliced at run time | |
| 普通層 | plain layer | `torch.nn` 的一般層 / an ordinary `torch.nn` layer | |

## 訓練 / Training

| 中文 | English | 定義 / Definition | 避免 / Avoid |
|---|---|---|---|
| base loss | base loss | base 模型的 loss / the loss of the base model | |
| 增強 loss | augmented loss | 增強網路的 loss，乘上 α 後加到 base loss / the loss of the augmented network, scaled by α and added to the base loss | aug loss |
| 寬度增強 | width augmentation | 加寬通道群組 / widen channel groups | |
| 深度增強 | depth augmentation | 在指定 block 後加入額外 block / add extra blocks after chosen blocks | |
| 寬度倍數 | width multiplier | 群組寬度相對 base 寬度的倍數 / a group's width relative to its base width | |
| 深度增強點 | depth point | 之後可加入額外 block 的 block / a block that may be followed by extra blocks | |
| 額外 block | extra block | 深度增強加入的 block 副本 / a copy of a block added by depth augmentation | |
| 彈性深度 | elastic depth | Once-for-All 的深度取樣方法 / the depth-sampling technique of Once-for-All | |
| 欠擬合 | underfitting | 模型連訓練資料都學不好 / the model cannot fit even the training data | |
| 過擬合 | overfitting | 模型記住訓練資料，但泛化差 / the model fits the training data but generalizes poorly | |
| 配方 | recipe | 訓練超參數與排程的組合 / a set of training hyperparameters and schedules | |

## 通道結構 / Channel structure

| 中文 | English | 定義 / Definition | 避免 / Avoid |
|---|---|---|---|
| 通道群組 | channel group | 寬度與順序必須相同的一組通道 / channels that must share width and order | |
| 固定群組 | fixed group | 保持原寬度、不排序的群組 / a group that keeps its original width and is not sorted | |
| 可增強群組 | free group | 可以加寬與排序的群組 / a group that can be widened and sorted | |
| 段 | segment | concat 後，一個群組在通道維度上占的一段 / the slice of the channel dimension that one group occupies after concat | |
| 白名單 | whitelist | 已知逐通道獨立運算的操作 / operations known to act on each channel independently | |
| 生產者 | producer | 輸出某群組的 dense 層 / the dense layer whose output is a group | 產生者 |
| 讀取者 | reader | 讀取某群組的層 / a layer that reads a group | |
| 追蹤 | tracing | 用 `torch.fx` 記錄模型的運算圖 / record the model's operation graph with `torch.fx` | |

## 排序、BN 與匯出 / Sorting, BN and export

| 中文 | English | 定義 / Definition | 避免 / Avoid |
|---|---|---|---|
| 通道排序 | channel sorting | 把重要的通道移到群組前段 / move important channels to the front of each group | |
| 重要度 | importance | 讀取者權重的 L1 範數 / L1 norm of the readers' weights | |
| 排列 | permutation | 通道的新順序 / a new order of the channels | |
| 權重共用 | weight sharing | 所有寬度使用同一組權重 / all widths use the same weights | |
| 綁定權重 | tied weights | 兩個層共用同一個參數張量 / two layers share one parameter tensor | |
| 殘差連接 | residual connection | `x + f(x)` | |
| 恆等函數 | identity function | 輸出等於輸入的函數 / a function whose output equals its input | |
| running 統計值 | running statistics | BN 在訓練中累積的平均與變異數 / the mean and variance that BN accumulates during training | |
| 重新校正 BN 統計值 | BN recalibration | 在指定寬度下重新估計 running 統計值 / re-estimate running statistics at a given width | |
| 匯出 | export | 產生不依賴本套件的普通 PyTorch 模型 / produce a plain PyTorch model that does not need this package | |

## 實驗 / Experiments

| 中文 | English | 定義 / Definition | 避免 / Avoid |
|---|---|---|---|
| 測試準確率 | test accuracy | 測試集的 top-1 準確率 / top-1 accuracy on the test set | |
| 訓練準確率 | train accuracy | 前 10,000 張訓練影像的 top-1 準確率，無資料增強 / top-1 accuracy on the first 10,000 training images, no data augmentation | |
| 配對差異 | paired difference | 同一 seed 的兩個方法相減 / difference between two methods with the same seed | |
| 官方實作 | official implementation | 論文作者（MIT HAN Lab）公開的程式 / the code released by the paper's authors (MIT HAN Lab) | |
