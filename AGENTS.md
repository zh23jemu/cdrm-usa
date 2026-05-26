# 项目维护记录

## 项目目标

本项目实现 CDRM-USA：面向 CWRU 轴承数据集的单源域泛化电机故障诊断框架。训练阶段只使用一个负载工况作为源域，评估阶段在其它负载工况上测试泛化性能。

## 技术栈

- Python + PyTorch：模型训练、损失计算与 GPU 推理。
- NumPy / SciPy：读取 `.mat` 信号与基础数值处理。
- scikit-learn：伪工况聚类、宏平均 F1、混淆矩阵和 t-SNE 可视化。
- PyYAML：集中式实验配置。
- Matplotlib：离线结果图与诊断图生成。

## 当前架构

- `train.py`：单次实验入口，负责读取配置、构建数据、模型、优化器、调度器、训练循环、验证和目标域评估。
- `run_all.py`：批量运行多方法、多源负载、多随机种子的实验，并聚合目标域指标。
- `scripts/`：提供单次训练和全量实验的 Slurm 提交脚本，默认从 `SLURM_SUBMIT_DIR` 回到提交目录运行。
- `configs/default.yaml`：数据窗口、STFT、训练超参、CDRM/USA 和 baseline 参数的默认配置。
- `data/`：CWRU 文件解析、滑窗切片、归一化、伪工况标签生成和 DataLoader 构建。
- `models/`：1D/2D backbone、CDRM、USA、组合模型和多种 baseline。
- `utils/`：随机种子、日志、损失函数和指标工具。
- `tools/visualize.py`：从 checkpoint 提取特征，生成 t-SNE 和混淆矩阵图。
- `CRWU/`：当前仓库内置的 CWRU 原始 `.mat` 数据。
- `results/figs/`：当前保留的小型结果图片，用于论文、报告或复现实验展示。

## 开发规范

- 默认使用项目本地 `.venv`，不要依赖系统 Python 或激活环境。
- 新增 Python 代码优先保持现有模块边界，避免无关重构。
- 训练输出中的 checkpoint、日志和临时产物不纳入版本管理；小型分析图可保留。
- 涉及长时间训练时优先使用 Slurm 脚本，GPU 默认使用 `aws` 分区、账号 `gpo-ifv7xx`、QOS `normal`。
- PyTorch GPU 环境默认使用 CUDA 12.x 兼容 wheel，优先选择 `cu126` 源。

## Current Status

已完成项目 GitHub Public 发布，并修复 Slurm 脚本在计算节点上错误使用 spool 路径导致输出目录无权限的问题。

## Recent Changes

- 新增项目级维护记录，记录项目目标、技术栈、架构和维护约束。
- 新增 `.gitignore`，忽略本地虚拟环境、Python 缓存、训练 checkpoint、训练日志、Slurm 输出和编辑器/系统文件。
- 新增 Slurm 训练脚本，便于在集群上运行默认 CDRM-USA 训练。
- 已创建 GitHub Public 仓库 `zh23jemu/cdrm-usa`，并推送当前 `master` 分支。
- 修复 `scripts/train_slurm.sbatch` 的工作目录逻辑，改为从 `SLURM_SUBMIT_DIR` 进入提交目录。
- 新增 `scripts/run_all_slurm.sbatch`，用于在 Slurm 上运行完整 `run_all.py` 实验。

## Next TODO

- 创建 `.venv` 并安装依赖后，执行一次短 epoch smoke test，确认数据路径、模型前向和训练循环可用。
- 进一步检查 `README.md` 的运行命令，必要时改为项目 `.venv` 形式。
- 检查 GitHub 仓库页面、默认分支和大文件展示是否正常。
- 在服务器执行 `git pull` 后，用 `EPOCHS=1 sbatch scripts/train_slurm.sbatch` 重新验证训练任务。

## Open Issues

- 当前尚未实际跑通训练；完整训练可能耗时较长，应优先在 GPU/Slurm 环境中验证。
- `CRWU/` 数据约 658MB，初版提交会包含数据集，远端仓库如有体积限制可能需要后续改用外部数据下载说明或 Git LFS。
- `README.md` 仍使用通用 `pip` / `python` 示例，与本项目维护规则中的 `.venv` 优先策略不完全一致。

## Architecture Decisions

- 保留 `CRWU/` 原始数据作为当前可复现实验输入，不在 `.gitignore` 中忽略 `.mat` 文件。
- 保留 `results/figs/` 的小型结果图片，但忽略 `results/ckpts/` 和 `results/logs/` 等可再生成运行态产物。
- 初版提交阶段不改动模型和训练逻辑，先建立清晰的版本边界与项目状态记录。
