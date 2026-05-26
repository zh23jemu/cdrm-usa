#!/bin/bash
#
# CDRM-USA 全量实验 Slurm 提交脚本。
# 默认运行 run_all.py，对多个方法和源负载进行批量训练与聚合评估。
# 如需临时切换分区，可在提交时使用：
#   sbatch --partition=目标分区 scripts/run_all_slurm.sbatch

#SBATCH --job-name=cdrm-run-all
#SBATCH --partition=aws
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=results/logs/slurm-%x-%j.out

set -euo pipefail

# 始终从 sbatch 提交时所在的项目根目录运行。
# 这样无论 Slurm 将脚本复制到哪个 spool 目录，数据、配置和输出路径都保持正确。
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Slurm 不会自动创建日志目录；这里创建的是运行态目录，不属于源码内容。
mkdir -p results/logs results/ckpts

# 使用项目本地虚拟环境，避免误用系统 Python。
PYTHON_BIN=".venv/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  echo "未找到 $PYTHON_BIN。请先创建 .venv 并安装 requirements.txt 中的依赖。"
  exit 1
fi

# 默认运行完整方法与源负载扫描。
# 可通过环境变量覆盖常用参数，例如：
#   EPOCHS=1 METHODS="cdrm_usa erm" SOURCES="0 1" sbatch scripts/run_all_slurm.sbatch
CONFIG="${CONFIG:-configs/default.yaml}"
EPOCHS="${EPOCHS:-}"
METHODS="${METHODS:-}"
SOURCES="${SOURCES:-}"
SEEDS="${SEEDS:-}"
OUT="${OUT:-results/all_methods.json}"

CMD=("$PYTHON_BIN" run_all.py --config "$CONFIG" --out "$OUT")
if [ -n "$EPOCHS" ]; then
  CMD+=(--epochs "$EPOCHS")
fi
if [ -n "$METHODS" ]; then
  # shellcheck disable=SC2206
  METHOD_ARGS=($METHODS)
  CMD+=(--methods "${METHOD_ARGS[@]}")
fi
if [ -n "$SOURCES" ]; then
  # shellcheck disable=SC2206
  SOURCE_ARGS=($SOURCES)
  CMD+=(--sources "${SOURCE_ARGS[@]}")
fi
if [ -n "$SEEDS" ]; then
  # shellcheck disable=SC2206
  SEED_ARGS=($SEEDS)
  CMD+=(--seeds "${SEED_ARGS[@]}")
fi

echo "运行命令: ${CMD[*]}"
"${CMD[@]}"
