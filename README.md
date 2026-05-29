# Efficient Bilevel Optimization for CKA-Guided MoE Upcycling

> Open-source release accompanying our ICML 2026 paper.
> Method codename: **`upcycle_cka_v5_v4`** (V5 algorithm with the V4
> hyperparameter configuration that produced the camera-ready numbers).

This release ships the **single training/inference path** that reproduces the
main Llama-3.2-1B Last Acc / BWT / expansion-rate numbers reported in the
paper. Everything that is not part of this path has been removed: competing
continual-learning baselines (EWC / LwF / GEM / OGD / O-LoRA / L2P / LoRA),
generative-replay variants (LFPT5 / MbPA++), and the orthogonal V7 expert-merge
exploration live in our internal research repository and are not part of this
release. The former V2/V3/V4 ablation modules have been consolidated into the
single `upcycling_cka_v5.py` trainer (the parent classes are still defined
there, in one file, so V5 can reuse their methods).

---

## 1. Repository layout

```
open_source/
|-- training/
|   |-- main.py                            # DeepSpeed entry point (V5-only CKA dispatch)
|   `-- params.py                          # CL method registry
|-- inference/
|   `-- infer_single.py                    # Per-checkpoint, per-round inference
|-- model/
|   |-- base_model.py                      # CL_Base_Model + SeqFT/save hooks
|   `-- Dynamic_network/
|       |-- upcycling_cka_v5.py            # MAIN ENTRY: full V2->V5 trainer (single file)
|       |-- upcycling_refactored.py        # Expert / Router / Upcycle primitives
|       `-- checkpoint_manager.py          # MoE-aware HF config writer
|-- evaluations/
|   |-- cl_metrics.py                      # Last Acc / Stage Acc / BWT
|   |-- representation_metrics.py
|   `-- cka_evaluator.py                   # Linear / differentiable CKA
|-- utils/                                 # data, ds_utils, model_utils, flash-attn
|-- scripts/
|   |-- train_upcycle_cka_v5_v4.sh         # one-command reproduction
|   |-- infer_parallel.sh                  # 4-GPU parallel inference
|   |-- visualize_inference_results.py     # extracts metrics + plots heatmaps
|   `-- prepare_blurry_data.py             # builds the Blurry Mixed-Domain stream
|-- requirements.txt
|-- environment.yml                        # full pinned conda export
|-- LICENSE
`-- README.md (this file)
```

`upcycling_cka_v5.py` defines the whole inheritance chain
`DifferentiableNasCKAV5 -> NasCKAUpcycleV4 -> AdaptiveCKAUpcycleV3 ->
AdaptiveCKAUpcycle -> BilevelCKAUpcycleV2 -> SimpleCKAUpcycleV2` in a single
file; the CLI only exposes `--cka-version v5`.

---

## 2. Environment

The reported runs used **Python 3.12, CUDA 12.x, PyTorch 2.9, DeepSpeed 0.15,
Transformers 4.57** on 4$\times$A100 80GB.

```bash
conda env create -f environment.yml -n trace
conda activate trace
# or use the lighter pip path:
pip install -r requirements.txt
```

Sanity check:

```bash
python -c "
import torch, deepspeed, transformers, accelerate, peft, datasets
print('torch       :', torch.__version__, '| CUDA:', torch.cuda.is_available(),
      '| devices:', torch.cuda.device_count())
print('deepspeed   :', deepspeed.__version__)
print('transformers:', transformers.__version__)
"
```

Expected: at least 4 visible CUDA devices for the default training config.

---

## 3. Data and base models

### TRACE benchmark (`LLM-CL-Benchmark_5000`)

Download from the original TRACE release and place at:

```
/data/datasets/TRACE-Benchmark/LLM-CL-Benchmark_5000/
|-- MeetingBank/        # task 0
|-- Py150/              # task 1
|-- NumGLUE-cm/         # task 2
|-- NumGLUE-ds/         # task 3
|-- 20Minuten/          # task 4
`-- C-STANCE/           # task 5
```

The 6-task sequence used in the paper is exactly:
`MeetingBank,Py150,NumGLUE-cm,NumGLUE-ds,20Minuten,C-STANCE`.
Override the location via `DATA_PATH=...` before launching.

### Blurry Mixed-Domain Stream (robustness ablation)

```bash
python scripts/prepare_blurry_data.py
```

Produces `/data/datasets/TRACE-Benchmark/Mixed_Chunks_5000/Chunk_{1..6}/{train,eval,test}.json`
with the 4000A+1000B sliding-window overlap described in the appendix.

### Base models

```
/data/models/Llama-3.2-1B-Instruct/    # default backbone
/data/models/Llama-3.2-3B-Instruct/    # appendix scaling
/data/models/Qwen3-0.6B/               # appendix cross-architecture
```

---

## 4. Reproduce the main result

```bash
conda activate trace
NUM_GPUS=4 CUDA_VISIBLE_DEVICES=0,1,2,3 \
BATCH_SIZE=8 GRAD_ACCUM=1 \
bash scripts/train_upcycle_cka_v5_v4.sh
```

Outputs are written under

```
<OUTPUT_BASE>/Llama-3.2-1B-Instruct/cl/upcycle_cka_v5/cka_v5_v4_<TIMESTAMP>/
|-- 0/  1/  2/  3/  4/  5/        # per-task HF checkpoints
|-- train.log
|-- train_command.sh              # exact command that produced this run
`-- v4_config.md                  # captured config summary
```

Expected numbers (Llama-3.2-1B, 4$\times$A100): Last Acc ~= 45.15, BWT ~= -3.71,
expansion rate ~= 65% (157/240).

### Ablations from the paper (env-var overrides only, never edit the script)

```bash
# NAS-only (no CKA guidance) - bidirectional decoupling
MASK_OVERRIDE=random LAMBDA_CKA=0 \
NUM_GPUS=4 CUDA_VISIBLE_DEVICES=0,1,2,3 BATCH_SIZE=8 GRAD_ACCUM=1 \
bash scripts/train_upcycle_cka_v5_v4.sh

# Counterfactual inverted CKA
MASK_OVERRIDE=invert \
NUM_GPUS=4 CUDA_VISIBLE_DEVICES=0,1,2,3 BATCH_SIZE=8 GRAD_ACCUM=1 \
bash scripts/train_upcycle_cka_v5_v4.sh

# Frozen inverted CKA
MASK_OVERRIDE=invert FREEZE_MASK=1 \
NUM_GPUS=4 CUDA_VISIBLE_DEVICES=0,1,2,3 BATCH_SIZE=8 GRAD_ACCUM=1 \
bash scripts/train_upcycle_cka_v5_v4.sh

# Blurry Mixed-Domain Stream
DATA_PATH=/data/datasets/TRACE-Benchmark/Mixed_Chunks_5000 \
TASKS=Chunk_1,Chunk_2,Chunk_3,Chunk_4,Chunk_5,Chunk_6 \
NUM_GPUS=4 CUDA_VISIBLE_DEVICES=0,1,2,3 BATCH_SIZE=8 GRAD_ACCUM=1 \
bash scripts/train_upcycle_cka_v5_v4.sh

# lambda_CKA sweep
for L in 0.01 0.1 1.0; do
  LAMBDA_CKA=$L NUM_GPUS=4 CUDA_VISIBLE_DEVICES=0,1,2,3 BATCH_SIZE=8 GRAD_ACCUM=1 \
  bash scripts/train_upcycle_cka_v5_v4.sh
done
```

`MASK_OVERRIDE` and `FREEZE_MASK` are env-only knobs read directly inside
`upcycling_cka_v5.py:_setup_supernet`.

### Cross-architecture (Qwen3-0.6B)

```bash
MODEL_PATH=/data/models/Qwen3-0.6B \
NUM_GPUS=4 CUDA_VISIBLE_DEVICES=0,1,2,3 BATCH_SIZE=8 GRAD_ACCUM=1 \
bash scripts/train_upcycle_cka_v5_v4.sh
```

---

## 5. Inference and metric extraction

```bash
RUN_DIR=$(ls -dt outputs_prototype/Llama-3.2-1B-Instruct/cl/upcycle_cka_v5/cka_v5_v4_* | head -1)

# Parallel inference over the 6 rounds, 4 GPUs
bash scripts/infer_parallel.sh "$RUN_DIR"

# Compute Last Acc / Stage Acc / BWT and dump heatmaps
python scripts/visualize_inference_results.py --results-dir "$RUN_DIR/inference_results"
```

`scripts/infer_parallel.sh` auto-detects the model path and number of experts
per task from `train_command.sh` inside the run directory and dispatches 6
round-level inferences across 4 GPUs, writing
`results-{round}-{task_idx}-{task_name}.json` files under `inference_results/`.

---

## 6. Citation

```bibtex
@inproceedings{yu2026bilevel,
  title     = {Efficient Bilevel Optimization for {CKA}-Guided {MoE} Upcycling},
  author    = {Yu, Zhiyuan and Yang, Enneng and Jiang, Hao and Zhu, Guojie and
               He, Feihong and Wang, Peng and Shen, Li},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

The TRACE benchmark this work builds on:

```bibtex
@article{wang2023trace,
  title   = {TRACE: A Comprehensive Benchmark for Continual Learning in Large Language Models},
  author  = {Wang, Xiao and Zhang, Yuansen and Chen, Tianze and others},
  journal = {arXiv preprint arXiv:2310.06762},
  year    = {2023}
}
```

---

## 7. Known gotchas

- **GPU memory.** The Supernet phase temporarily holds both old and new
  candidate experts. Peak VRAM is ~29 GB/GPU on A100-80GB at `BATCH_SIZE=8`;
  lower `BATCH_SIZE` and raise `GRAD_ACCUM` proportionally on smaller cards
  (target effective batch = 32).
- **`MASK_OVERRIDE` is env-only.** Read in `upcycling_cka_v5.py:_setup_supernet`;
  if you forget to export it, the run silently uses CKA initialization.
- **Inference uses the original TRACE test sets even for the blurry stream.**
  This is correct - we want a fair comparison against the clean-boundary
  baseline on the same held-out data.
- **Method2Class registry is intentionally minimal.** Only `base` and
  `upcycle` are registered in `params.py`; the CKA-guided V5 trainer is
  dispatched from `training/main.py` when `--cka-regularization
  --cka-version v5` is set. All competing baselines (EWC / LwF / GEM / OGD /
  O-LoRA / L2P / LoRA / MbPA++ / LFPT5) and the Drop-Upcycling / BTM MoE
  baselines have been removed; if you need them, see the full research repo.
