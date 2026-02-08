# 使用 HF-Mirror 镜像站下载数据集

由于网络访问限制，代码已支持使用 [HF-Mirror](https://hf-mirror.com) 镜像站来下载 HuggingFace 数据集和模型。

## 使用方法

### 方法一：设置环境变量（推荐）

在运行脚本前设置环境变量：

**Linux/Mac:**
```bash
export HF_MIRROR=1
python -m scripts.base_train
```

**Windows PowerShell:**
```powershell
$env:HF_MIRROR = "1"
python -m scripts.base_train
```

**Windows CMD:**
```cmd
set HF_MIRROR=1
python -m scripts.base_train
```

### 方法二：直接设置 HF_ENDPOINT

也可以直接设置 `HF_ENDPOINT` 环境变量：

**Linux/Mac:**
```bash
export HF_ENDPOINT=https://hf-mirror.com
python -m scripts.base_train
```

**Windows PowerShell:**
```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
python -m scripts.base_train
```

### 方法三：永久设置（推荐用于开发环境）

将环境变量添加到您的 shell 配置文件中：

**Linux/Mac (~/.bashrc 或 ~/.zshrc):**
```bash
export HF_MIRROR=1
# 或者
export HF_ENDPOINT=https://hf-mirror.com
```

**Windows:**
在系统环境变量中添加 `HF_MIRROR=1` 或 `HF_ENDPOINT=https://hf-mirror.com`

## 代码自动支持

代码已自动支持以下功能：

1. **数据集下载** (`nanochat/dataset.py`)
   - 自动检测 `HF_ENDPOINT` 或 `HF_MIRROR` 环境变量
   - 使用镜像站下载 parquet 文件

2. **HuggingFace datasets 库**
   - 所有使用 `load_dataset()` 的地方（如 tasks/arc.py, tasks/mmlu.py 等）
   - 会自动使用 `HF_ENDPOINT` 环境变量

## 支持的数据集

以下数据集下载会自动使用镜像站：

- **训练数据集**: `karpathy/fineweb-edu-100b-shuffle` (parquet 文件)
- **评估数据集**:
  - `allenai/ai2_arc` (ARC)
  - `cais/mmlu` (MMLU)
  - `openai/openai_humaneval` (HumanEval)
  - `openai/gsm8k` (GSM8K)
  - `HuggingFaceTB/smol-smoltalk` (SmolTalk)

## 验证是否生效

运行训练脚本时，如果看到以下日志，说明正在使用镜像站：

```
Using HuggingFace mirror: https://hf-mirror.com
```

或者检查下载的 URL，应该显示 `hf-mirror.com` 而不是 `huggingface.co`。

## 故障排除

### 1. 仍然无法下载

如果设置了环境变量但仍然无法下载，请检查：

```bash
# 检查环境变量是否设置
echo $HF_MIRROR
echo $HF_ENDPOINT

# 在 Python 中检查
python -c "import os; print('HF_MIRROR:', os.environ.get('HF_MIRROR')); print('HF_ENDPOINT:', os.environ.get('HF_ENDPOINT'))"
```

### 2. 部分数据集仍然失败

某些数据集可能有内置的下载脚本，需要手动修改。如果遇到问题，可以：

1. 手动从 [HF-Mirror 网站](https://hf-mirror.com) 下载
2. 使用 `huggingface-cli` 工具：

```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --repo-type dataset karpathy/fineweb-edu-100b-shuffle --local-dir ./data
```

### 3. 需要认证的数据集

对于需要登录的 Gated Repo，请参考 [HF-Mirror 官方文档](https://hf-mirror.com) 中的认证说明。

## 参考资源

- [HF-Mirror 官网](https://hf-mirror.com)
- [HF-Mirror 使用教程](https://hf-mirror.com) (网站上有详细教程链接)
