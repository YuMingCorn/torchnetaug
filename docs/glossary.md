# 術語對照 / Glossary

中文文件與英文文件使用同一組術語。程式識別字（`NetAugModel`、`set_base()` 等）兩種語言都不翻譯。

Chinese and English documents use the same terms. Code identifiers (`NetAugModel`, `set_base()`, ...) are never translated.

| 中文 | English | 說明 / Meaning |
|---|---|---|
| 網路增強 | Network Augmentation (NetAug) | 訓練時把小模型嵌入較大的模型一起訓練 / train the tiny model inside larger models that share its weights |
| 小模型、base 模型 | tiny model, base model | 要部署的原始模型 / the original model that is deployed |
| 增強網路 | augmented network | 包含 base 模型的較寬或較深網路 / a wider or deeper network that contains the base model |
| 欠擬合 | underfitting | |
| 過擬合 | overfitting | |
| 寬度增強 | width augmentation | |
| 深度增強 | depth augmentation | |
| 寬度倍數 | width multiplier | `width_mults` |
| 深度增強點 | depth point | `depth=` 的一個鍵：之後可加額外 block 的 block / a key of `depth=`: a block that may be followed by extra blocks |
| 額外 block | extra block | 深度增強加入的 block 副本 / a copy of the block added by depth augmentation |
| 彈性深度 | elastic depth | Once-for-All 的做法 / the Once-for-All technique |
| 通道群組 | channel group | 必須同寬度、同順序的一組通道 / channels that must share width and order |
| 固定群組 | fixed group | 不增強的群組 / a group that keeps its original width |
| 可增強群組 | free group | |
| 段 | segment | concat 後，一個群組在通道維度上占的一段 / the slice of the channel dimension that one group occupies after concat |
| 白名單 | whitelist | 已知會逐通道獨立運算的操作 / operations known to act on each channel independently |
| 動態層 | dynamic layer | 以最大寬度存權重、執行時切片的層 / a layer stored at max width and sliced at run time |
| 普通層 | plain layer | `torch.nn` 的一般層 / an ordinary `torch.nn` layer |
| 生產者 | producer | 輸出某群組的 dense 層 / the dense layer whose output is a group |
| 讀取者 | reader | 讀取某群組的層 / a layer that reads a group |
| 通道排序 | channel sorting | |
| 重要度 | importance | 讀取者權重的 L1 範數 / L1 norm of the readers' weights |
| 排列 | permutation | |
| 權重共用 | weight sharing | |
| 綁定權重 | tied weights | |
| 殘差連接 | residual connection | |
| 恆等函數 | identity function | |
| 重新校正 BN 統計值 | BN recalibration | `recalibrate_bn` |
| running 統計值 | running statistics | BN 的 `running_mean` / `running_var` |
| 匯出 | export | 產生不依賴本套件的普通 PyTorch 模型 / produce a plain PyTorch model that does not need this package |
| 追蹤 | tracing | `torch.fx` 符號追蹤 / `torch.fx` symbolic tracing |
| 配對差異 | paired difference | 同一 seed 的兩個方法相減 / difference between two methods with the same seed |
| 測試準確率 | test accuracy | top-1 |
| 訓練準確率 | train accuracy | 前 10,000 張訓練影像，無資料增強 / first 10,000 training images, no data augmentation |
| 參考實作 | reference implementation | 本專案作者先前的 YOLOv8 實作 `netaug/` / the author's earlier YOLOv8 implementation |
| 官方實作 | official implementation | MIT HAN Lab 的程式 / the MIT HAN Lab code |
