# neural_network_augmentation

目標：建立一個 PyTorch 神經網路增強（Network Augmentation, NetAug）套件。

## 硬性規則

- **`netaug/` 是唯讀的舊資料（不進 git），不可修改或重新格式化，也不可在任何文件中提及。**
  - 不要在 `netaug/` 裡執行會寫檔的指令（包含 `python -c "import netaug"`，它會產生 `.pyc`）。
  - 新套件的程式碼放在別的目錄，不要從 `netaug` 匯入。

## 已決定（2026-11-18）

- 專案已於 2026-11-24 結束：測試顯示實作符合論文的定義，但 NetAug 在本專案的 CIFAR 實驗中沒有顯著效果（見 summary.md）。

- 通用的純 PyTorch 套件 `torchnetaug`（`src/torchnetaug/`），不依賴 ultralytics。
- MIT 授權。不可從 `netaug/` 複製程式碼（它帶 Ultralytics GPL-3.0 檔頭）。
- Python 與 PyTorch 用最新版。開發環境：Python 3.14、torch 2.11+cu128（顯卡驅動 576 最高只支援 CUDA 12.9，而較新的 torch 只有 cu130 版）。PyPI 最新的 torch 2.15.0（CPU）也已跑過全部測試（2026-11-19）：`uv run --isolated --no-project --no-sources --with "torch==2.15.0" --with torchvision --with pytest --with-editable . pytest`。

## 套件與驗證

- 專案總結與未來工作：[summary.md](summary.md)；程式碼設計與正確性驗證：[docs/design.md](docs/design.md)；用法：[README.md](README.md)；實驗結果與觀察：[docs/experiments.md](docs/experiments.md)；術語對照：[docs/glossary.md](docs/glossary.md)。
- 文件中英並列：中文是 `X.md`，英文是同目錄的 `X.en.md`，兩者開頭互相連結。改中文就要同步改英文。
- 文件風格（2026-11-23 決定）：用語依 glossary 的「避免」欄，不用同義詞；每份文件以最多 5 點的「重點」開頭，清單不超過 5 點；流程用 Mermaid 圖。
- `uv run pytest`：正確性測試，以及 `tests/test_docs.py`（`DOCS` 清單中每份文件的 python 區塊都必須能執行，含英文版）。改了 API 就要同步改文件。ONNX 測試需要開發依賴 onnx、onnxscript、onnxruntime，沒裝時會跳過。
- 增強維度：寬度（論文）與深度（`depth=`，參考 Once-for-All elastic depth；官方實作沒有深度增強）。
- `NetAugModel` 只接受普通 `torch.nn` 層組成的模型；含動態層會丟 `TypeError`（2026-11-19 決定，替代方案見 design.md 第 11 節）。
- 效果實驗：`examples/cifar_netaug.py`（`--depth` 為深度增強），資料放在 `data/`（不進 git）；`examples/summarize.py` 可同時讀多個 jsonl。

## 參考資料

- [docs/reference/netaug_paper.md](docs/reference/netaug_paper.md)：NetAug 論文（ICLR 2022）與官方實作重點，以及兩者的對照表。

## 環境

- Windows 10，RTX 3060 Ti（8 GB）。系統沒有 Python，一律用 `uv run ...`（虛擬環境在 `.venv/`）。
- CIFAR 官方主機的 SSL 憑證過期，Python 下載會失敗；用 `curl -L` 下載到 `data/`，torchvision 會檢查 MD5。
