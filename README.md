---
title: Yangtez Beta
emoji: ⚡
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 6.11.0
app_file: app.py
pinned: false
license: mit
short_description: fund porposal reviewer
---

Proposal / fund-application review UI (Gradio). Space configuration: [Hugging Face Spaces config](https://huggingface.co/docs/hub/spaces-config-reference).

## 本地运行（推荐流程）

固定使用**同一个** Python 环境，避免「包装在 A 环境、却在 B 环境跑 `app.py`」。

### 1. 固定环境（二选一）

- **使用 Conda 环境（推荐）**  
  每次新开终端后先执行（环境名以你本机为准，例如 `yangtze`）：

  ```bash
  conda activate yangtze
  ```

- **不使用 Conda**  
  始终用同一个 `python` 可执行文件路径即可。

### 2. 进入项目目录

```bash
cd C:\Users\ubcas\Desktop\Yangtez-beta
```

（路径请按你本机仓库位置修改。）

### 3. 安装依赖

**新环境**或 **`requirements.txt` 有更新**时执行一次即可，不必每次启动都装：

```bash
python -m pip install -r requirements.txt
```

### 4. 配置环境变量

在仓库根目录创建 `.env`（勿提交到公开仓库），至少包含：

```env
OPENAI_API_KEY=你的密钥
```

可选：`TESSERACT_CMD`（Tesseract 未加入系统 PATH 时，指向 `tesseract.exe` 的完整路径）、`REVIEW_OUTPUT_LANG`（`zh` / `en`）、`HF_TOKEN`（从 Hugging Face Hub 拉模型时提高限额）等。

若 OpenAI 出现 **429 / TPM 限流**：Stage 0 的 Vision 页与质量自检会自动退避重试。也可加大间隔，例如在 `.env` 中设置 `VISION_LLM_PAGE_GAP_SEC=0.6`、`VISION_LLM_MAX_RETRIES=12`；或暂时关闭 Vision：`ENABLE_VISION_LLM=0`（仍保留 OCR / pdfplumber / Table Transformer 路径）。

### 5. 启动应用

```bash
python app.py
```

在浏览器中打开终端里显示的本地地址（例如 `http://127.0.0.1:7862`）。

### 原则

在同一终端会话中：**先 `conda activate`（若使用 Conda）→ 再 `cd` 到项目目录 → 再 `python app.py`**，保证 `pip` 安装的包与运行使用的是同一解释器。

### 自检（可选）

若怀疑环境不对，可在**已激活环境且已 `cd` 到项目后**执行：

```bash
python -c "import timm, gradio; print('ok')"
```

若报错，说明当前终端使用的不是你以为的那个环境；请回到第 1 步切换/激活正确环境后，再执行第 3 步安装依赖。

### OCR / 版式（Windows 提示）

- PDF 文本层很薄时，会尝试 **Tesseract OCR**；未安装或未在 PATH 时，程序会尝试常见安装路径（如 `C:\Program Files\Tesseract-OCR\tesseract.exe`），也可在 `.env` 中设置 `TESSERACT_CMD`。
- Stage 0 版式中的 **Table Transformer** 依赖 **`timm`**（已在 `requirements.txt`）。若未安装，表格检测会回退到 **pdfplumber**，不影响整体评审流程。
