#!/bin/bash

# This script is configured to train your own GPT-2 grade LLM (pretraining + finetuning)
# It is designed to run on a blank 8XH100 GPU node and takes approximately 3 hours to complete.

# 1) Example launch (simplest):
# bash runs/speedrun.sh
# 2) Example launch in a screen session (because the run takes ~3 hours):
# screen -L -Logfile runs/speedrun.log -S speedrun bash runs/speedrun.sh
# 3) Example launch with wandb logging, but see below for setting up wandb first:
# WANDB_RUN=speedrun screen -L -Logfile runs/speedrun.log -S speedrun bash runs/speedrun.sh

# Default intermediate artifacts directory
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="/work/home/hf_cache/nanochat"
export UV_HOME="/work/home/hf_cache/uv"
export NANOCHAT_DATA_DIR="/work/home/hf_cache/nano_datasets"
# For NPU: Optionally set visible NPU devices (e.g., export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3)
# If not set, all available NPUs will be used
# For China network: Use HF-Mirror to download datasets (uncomment if needed)
# export HF_MIRROR=1
mkdir -p $NANOCHAT_BASE_DIR
mkdir -p $UV_HOME
mkdir -p $NANOCHAT_DATA_DIR

# -----------------------------------------------------------------------------
# Python setup with uv (no virtual environment)

# install uv to custom directory (if not already installed)
if ! command -v uv &> /dev/null; then
    export CARGO_HOME="$UV_HOME/cargo"
    export RUSTUP_HOME="$UV_HOME/rustup"
    # uv installer - install to custom location
    # The installer respects CARGO_HOME for installation location
    # Add retry logic for network issues
    curl -LsSf --retry 3 --retry-delay 2 https://astral.sh/uv/install.sh | CARGO_HOME="$CARGO_HOME" RUSTUP_HOME="$RUSTUP_HOME" sh || {
        echo "Warning: curl download had SSL issues, but installation may have succeeded."
        echo "Checking if uv was installed anyway..."
    }
    # Add uv to PATH - check multiple possible locations
    if [ -f "$CARGO_HOME/bin/uv" ]; then
        export PATH="$CARGO_HOME/bin:$PATH"
    elif [ -f "$HOME/.cargo/bin/uv" ]; then
        export PATH="$HOME/.cargo/bin:$PATH"
    elif [ -f "$HOME/.local/bin/uv" ]; then
        export PATH="$HOME/.local/bin:$PATH"
    fi
    # Also check if uv was installed to UV_HOME directly
    if [ -f "$UV_HOME/bin/uv" ]; then
        export PATH="$UV_HOME/bin:$PATH"
    fi
fi

# Verify uv is available
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not available after installation."
    echo "Checked paths: $CARGO_HOME/bin, $HOME/.cargo/bin, $HOME/.local/bin, $UV_HOME/bin"
    exit 1
fi

echo "Using uv from: $(which uv)"

# install the repo dependencies directly to system Python (no venv)
# Use --system flag to install to system Python instead of creating venv
# Note: uv may fail if there are packages with non-standard version formats (e.g., apex-0.1_ascend)
# In that case, we fall back to pip
if ! uv pip install --system -e ".[gpu]" 2>&1 | tee /tmp/uv_install.log; then
    echo "Warning: uv install failed (possibly due to non-standard package metadata like apex-0.1_ascend)."
    echo "Falling back to pip..."
    # Use pip3 if available, otherwise pip
    if command -v pip3 &> /dev/null; then
        pip3 install -e ".[gpu]"
    else
        pip install -e ".[gpu]"
    fi
fi

# -----------------------------------------------------------------------------
# wandb setup
# If you wish to use wandb for logging (it's nice!, recommended).
# 1) Make sure to first log in to wandb, e.g. run:
#    `wandb login`
# 2) Set the WANDB_RUN environment variable when running this script, e.g.:
#    `WANDB_RUN=d26 bash speedrun.sh`
if [ -z "$WANDB_RUN" ]; then
    # by default use "dummy" : it's handled as a special case, skips logging to wandb
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# During the course of the run, we will be writing markdown reports to the report/
# directory in the base dir. This command clears it out and writes a header section
# with a bunch of system info and a timestamp that marks the start of the run.
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer

# Download the first ~2B characters of pretraining dataset
# each data shard is ~250M chars
# so we download 2e9 / 250e6 = 8 data shards at this point
# each shard is ~100MB of text (compressed), so this is about ~800MB of data on disk
# look at dev/repackage_data_reference.py for details on how this data was prepared
python -m nanochat.dataset -n 8
# Immediately also kick off downloading more shards in the background while tokenizer trains
# Approximately 350 shards are needed for 10B tokens of data for pretraining.
# The maximum total number of shards available in the entire dataset is 1822.
python -m nanochat.dataset -n 370 &
DATASET_DOWNLOAD_PID=$!
# train the tokenizer with vocab size 2**15 = 32768 on ~2B characters of data
python -m scripts.tok_train
# evaluate the tokenizer (report compression ratio etc.)
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining)
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# d24 model (slightly overtrained is enough to beat GPT-2 => increase data:params ratio from compute optimal 10.5 (default) to 12)
# Note: --fp8 removed for NPU compatibility (FP8 only supported on CUDA H100+)
# Note: Using 4 NPUs instead of 8 GPUs
torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- --depth=26 --target-param-data-ratio=8.25 --device-batch-size=16 --device-type=npu --run=$WANDB_RUN
# evaluate the model: CORE metric, BPB on train/val, and draw samples
torchrun --standalone --nproc_per_node=4 -m scripts.base_eval -- --device-batch-size=16 --device-type=npu

# -----------------------------------------------------------------------------
# SFT (teach the model conversation special tokens, tool use, multiple choice)

# download 2.3MB of synthetic identity conversations to impart a personality to nanochat
# see dev/gen_synthetic_data.py for details on how this data was prepared and to get a sense of how you can easily tune it
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# run SFT and eval the model
torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft -- --device-batch-size=16 --device-type=npu --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=4 -m scripts.chat_eval -- -i sft --device-type=npu

# chat with the model over CLI! Leave out the -p to chat interactively
# python -m scripts.chat_cli -p "Why is the sky blue?"

# even better, chat with your model over a pretty WebUI ChatGPT style
# python -m scripts.chat_web

# -----------------------------------------------------------------------------
# Generate the full report by putting together all the sections
# report.md is the output and will be copied to current directory for convenience
python -m nanochat.report generate
