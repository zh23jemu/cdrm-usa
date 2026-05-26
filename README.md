# CDRM-USA: Causal Disentangled Representation Learning with Uncertainty-aware Structural Augmentation for Single-Source Domain-Generalized Motor Fault Diagnosis

This repository implements the proposed framework for motor fault diagnosis under the **single-source domain generalization (SS-DG)** setting on the **CWRU bearing dataset**, where the target working conditions are completely inaccessible during training.

## Highlights

1. **CDRM** (Causal Disentangled Representation Module): a working-condition-aware causal disentanglement module that splits the latent representation into a fault-invariant branch `z_f` and a working-condition branch `z_c`. Disentanglement is enforced with a Gradient Reversal Layer (GRL) on `z_f` plus an auxiliary condition classifier on `z_c` using pseudo working-condition labels (spectrum-based clustering + augmentation-aware indicators).

2. **USA** (Uncertainty-aware Structural Augmentation): a sparse-attention time-frequency encoder produces an attention map whose **per-region entropy** is treated as **structural uncertainty**. High-uncertainty (time, frequency) regions receive structural perturbations (frequency-band masking, time-slice dropout, local spectral perturbation), while low-uncertainty regions remain untouched. An uncertainty-weighted consistency loss is applied to enforce robust predictions across original and perturbed signals.

3. **Baselines (2024-2026)**: ERM, DANN, MixStyle, RSC, IRM, Fishr, SAGM, PCL, WDCNN, with a unified training pipeline.

## Project Layout

```
.
|-- CRWU/                       # CWRU raw data (provided)
|-- configs/default.yaml        # main config
|-- data/
|   |-- cwru.py                 # CWRU parser, sliding-window dataset, pseudo working-condition labels
|   |-- preprocess.py           # STFT, signal augmentations
|-- models/
|   |-- backbones.py            # 1D ResNet, 2D TF-ConvNeXt-tiny encoder
|   |-- cdrm.py                 # CDRM module
|   |-- usa.py                  # USA module
|   |-- full_model.py           # CDRM + USA composition
|   |-- baselines.py            # ERM, DANN, MixStyle, RSC, IRM, Fishr, SAGM, PCL, WDCNN
|-- utils/
|   |-- losses.py               # GRL, consistency, sparsity losses
|   |-- metrics.py              # accuracy, macro-F1, per-class accuracy
|   |-- seed.py
|   |-- logger.py
|-- train.py                    # single experiment entry
|-- run_all.py                  # cross-load DG sweep (4 source -> 3 target settings)
|-- requirements.txt
```

## Run

```
pip install -r requirements.txt
python train.py --config configs/default.yaml --method cdrm_usa --source 0
python run_all.py --config configs/default.yaml
```

GPU is used automatically when available (`torch.cuda.is_available()`).
