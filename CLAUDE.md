# neural_network_augmentation

目標：建立一個 PyTorch 神經網路增強（Network Augmentation, NetAug）套件。

## 硬性規則

- **`netaug/` 是唯讀參考資料，不可修改、移動、刪除或重新格式化。**
  - 這是作者 2023 年在 ultralytics YOLOv8 上做的 NetAug 實作。
  - 不要在 `netaug/` 裡執行會寫檔的指令（包含 `python -c "import netaug"`，它會產生 `.pyc`）。
  - 新套件的程式碼放在別的目錄，不要從 `netaug` 匯入。

## 已決定（2026-11-18）

- 通用的純 PyTorch 套件 `torchnetaug`（`src/torchnetaug/`），不依賴 ultralytics。
- MIT 授權。不可從 `netaug/` 複製程式碼（它帶 Ultralytics GPL-3.0 檔頭）。
- Python 與 PyTorch 用最新版。開發環境：Python 3.14、torch 2.11+cu128（顯卡驅動 576 最高只支援 CUDA 12.9，而較新的 torch 只有 cu130 版）。PyPI 最新的 torch 2.15.0（CPU）也已跑過全部測試（2026-11-19）：`uv run --isolated --no-project --no-sources --with "torch==2.15.0" --with torchvision --with pytest --with-editable . pytest`。

## 套件與驗證

- 架構與排序方式：[docs/architecture.md](docs/architecture.md)；用法：[README.md](README.md)。
- `uv run pytest`：正確性測試，以及 `tests/test_docs.py`（README 與 architecture.md 裡的 python 區塊都必須能執行）。改了 API 就要同步改文件。
- 效果實驗：`examples/cifar_netaug.py`，資料放在 `data/`（不進 git）。

## 參考資料

- [docs/reference/netaug_reference_impl.md](docs/reference/netaug_reference_impl.md)：`netaug/` 的結構、機制、兩種訓練演算法、已知問題。
- [docs/reference/netaug_paper.md](docs/reference/netaug_paper.md)：NetAug 論文（ICLR 2022）與官方程式重點，以及三者差異表。

## 環境

- Windows 10，RTX 3060 Ti（8 GB）。系統沒有 Python，一律用 `uv run ...`（虛擬環境在 `.venv/`）。
- CIFAR 官方主機的 SSL 憑證過期，Python 下載會失敗；用 `curl -L` 下載到 `data/`，torchvision 會檢查 MD5。
