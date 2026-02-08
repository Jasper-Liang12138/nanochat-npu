# 自定义路径配置说明

代码已修改以支持自定义路径，适用于天翼云GPU主机等环境。

## 环境变量配置

脚本 `runs/speedrun.sh` 已配置以下环境变量：

1. **UV 安装目录**: `/work/home/hf_cache/uv`
   - 通过 `UV_HOME` 环境变量设置
   - uv 会安装到 `$UV_HOME/cargo/bin/uv`

2. **数据集下载目录**: `/work/home/hf_cache/nano_datasets`
   - 通过 `NANOCHAT_DATA_DIR` 环境变量设置
   - 所有 parquet 文件会下载到此目录

3. **其他缓存目录**: `/work/home/hf_cache/nanochat`
   - 通过 `NANOCHAT_BASE_DIR` 环境变量设置
   - 用于存储 tokenizer、checkpoints 等

## 使用方法

直接运行脚本即可，无需额外配置：

```bash
export HF_ENDPOINT=https://hf-mirror.com
bash runs/speedrun.sh
```

## 修改说明

### 1. 移除了虚拟环境

- 不再创建 `.venv` 目录
- 使用 `uv pip install --system` 直接安装到系统 Python
- 直接使用系统 Python 运行脚本

### 2. 自定义 uv 安装路径

- uv 安装到 `/work/home/hf_cache/uv`
- 自动添加到 PATH 环境变量

### 3. 自定义数据集目录

- 数据集下载到 `/work/home/hf_cache/nano_datasets`
- 通过 `NANOCHAT_DATA_DIR` 环境变量控制

## 目录结构

运行后，目录结构如下：

```
/work/home/hf_cache/
├── uv/                    # uv 工具安装目录
│   └── cargo/
│       └── bin/
│           └── uv
├── nano_datasets/          # 数据集下载目录
│   ├── shard_00000.parquet
│   ├── shard_00001.parquet
│   └── ...
└── nanochat/               # 其他缓存文件
    ├── base_checkpoints/
    ├── report/
    └── ...
```

## 注意事项

1. **权限问题**: 确保 `/work/home/hf_cache/` 目录有写权限
2. **磁盘空间**: 数据集会占用较大空间，确保有足够磁盘空间
3. **Python 版本**: 确保系统 Python 版本 >= 3.9（代码已修改支持 Python 3.9）
4. **依赖安装**: 如果系统 Python 已有部分依赖，可能会跳过安装

## 故障排除

### uv 未找到

如果提示 `uv: 未找到命令`，检查：

```bash
# 检查 uv 是否安装
ls -la /work/home/hf_cache/uv/cargo/bin/uv

# 手动添加到 PATH
export PATH="/work/home/hf_cache/uv/cargo/bin:$PATH"
```

### 数据集下载失败

检查环境变量：

```bash
echo $NANOCHAT_DATA_DIR
echo $HF_ENDPOINT
```

### 权限错误

确保目录有写权限：

```bash
mkdir -p /work/home/hf_cache/{uv,nano_datasets,nanochat}
chmod -R 755 /work/home/hf_cache
```
