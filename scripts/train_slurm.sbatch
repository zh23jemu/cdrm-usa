#!/bin/bash
#
# CDRM-USA 单次训练 Slurm 提交脚本。
# 默认面向 GPU 训练任务：使用 aws 分区、gpo-ifv7xx 账号和 normal QOS。
# 如需临时切换分区，可在提交时使用：
#   sbatch --partition=目标分区 scripts/train_slurm.sbatch

#SBATCH --job-name=cdrm-usa-src0
#SBATCH --partition=aws
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=results/logs/slurm-%x-%j.out

set -euo pipefail

# 始终从项目根目录运行，确保配置文件、数据目录和输出目录路径一致。
cd "$(dirname "$0")/.."

# Slurm 不会自动创建日志目录；这里创建的是运行态目录，不属于源码内容。
mkdir -p results/logs results/ckpts

# 使用项目本地虚拟环境，避免误用系统 Python。
PYTHON_BIN=".venv/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  echo "未找到 $PYTHON_BIN。请先创建 .venv 并安装 requirements.txt 中的依赖。"
  exit 1
fi

# 默认运行 CDRM-USA 在 source load 0 上的单源域泛化训练。
# 可通过环境变量覆盖常用参数，例如：
#   METHOD=erm SOURCE=1 EPOCHS=10 sbatch scripts/train_slurm.sbatch
METHOD="${METHOD:-cdrm_usa}"
SOURCE="${SOURCE:-0}"
CONFIG="${CONFIG:-configs/default.yaml}"
EPOCHS="${EPOCHS:-}"
TAG="${TAG:-${METHOD}_src${SOURCE}_slurm}"

CMD=("$PYTHON_BIN" train.py --config "$CONFIG" --method "$METHOD" --source "$SOURCE" --tag "$TAG")
if [ -n "$EPOCHS" ]; then
  CMD+=(--epochs "$EPOCHS")
fi

echo "运行命令: ${CMD[*]}"
"${CMD[@]}"
