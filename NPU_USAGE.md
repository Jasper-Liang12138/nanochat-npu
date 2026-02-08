# 在华为昇腾NPU 910B上运行nanochat

本文档说明如何在4张华为昇腾NPU 910B上运行nanochat代码。

## 前置要求

1. **安装CANN工具链**
   - 确保已安装华为CANN（Compute Architecture for Neural Networks）工具链
   - CANN版本需要与PyTorch版本兼容

2. **安装PyTorch和torch_npu**
   ```bash
   # 安装支持NPU的PyTorch版本
   # 请根据您的CANN版本选择对应的PyTorch版本
   pip install torch_npu
   ```

3. **环境变量设置**
   ```bash
   # 设置可见的NPU设备（4张卡）
   export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
   ```

## 使用方法

### 1. 单卡训练/推理

```bash
# 自动检测NPU
python -m scripts.base_train --device-type=npu

# 或者让系统自动检测（NPU优先级高于CUDA）
python -m scripts.base_train
```

### 2. 4卡分布式训练

使用`torchrun`启动分布式训练：

```bash
# 4卡分布式训练
torchrun --nproc_per_node=4 -m scripts.base_train --device-type=npu

# 或者使用环境变量指定设备
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 -m scripts.base_train --device-type=npu
```

### 3. Web界面推理

```bash
# 启动Web服务器，使用4张NPU
python -m scripts.chat_web --device-type=npu --num-gpus=4
```

### 4. 命令行聊天

```bash
python -m scripts.chat_cli --device-type=npu
```

## 主要修改说明

代码已进行以下修改以支持昇腾NPU：

1. **设备检测** (`nanochat/common.py`)
   - 添加了NPU设备自动检测
   - NPU检测优先级：NPU > CUDA > MPS > CPU

2. **分布式训练**
   - NPU使用`hccl`后端（华为集合通信库）
   - CUDA使用`nccl`后端

3. **设备操作**
   - 添加了NPU的同步、内存查询等辅助函数
   - 支持NPU的autocast（混合精度训练）

4. **脚本更新**
   - 所有脚本的`--device-type`参数现在支持`npu`选项
   - 更新了设备相关的检查和操作

## 注意事项

1. **FP8训练**
   - 目前FP8训练仅支持CUDA，NPU上会自动忽略`--fp8`标志

2. **Flash Attention**
   - Flash Attention 3目前仅支持CUDA（Hopper GPU）
   - NPU上会使用PyTorch SDPA作为fallback

3. **性能优化**
   - 确保CANN驱动和固件版本是最新的
   - 根据实际硬件调整batch size等超参数

4. **内存管理**
   - NPU的内存管理可能与CUDA有所不同
   - 如果遇到OOM，可以减小`--device-batch-size`

## 故障排除

1. **无法检测到NPU**
   ```bash
   # 检查torch_npu是否正确安装
   python -c "import torch_npu; print(torch_npu.is_available())"
   ```

2. **分布式训练失败**
   - 确保所有NPU设备可见：`echo $ASCEND_RT_VISIBLE_DEVICES`
   - 检查CANN环境是否正确配置

3. **性能问题**
   - 检查NPU利用率：使用`npu-smi`工具
   - 确保数据加载没有成为瓶颈

## 示例命令

完整的4卡训练示例：

```bash
# 设置环境变量
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

# 启动4卡分布式训练
torchrun --nproc_per_node=4 -m scripts.base_train \
    --device-type=npu \
    --depth=12 \
    --max-seq-len=2048 \
    --device-batch-size=32 \
    --total-batch-size=524288
```

训练完成后启动Web界面：

```bash
python -m scripts.chat_web --device-type=npu --num-gpus=4
```

## 参考资源

- [华为昇腾NPU官方文档](https://www.hiascend.com/)
- [PyTorch NPU支持](https://gitee.com/ascend/pytorch)
