#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import sys
sys.dont_write_bytecode = True

import argparse
import os
import math
import sys
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoModelForCausalLM,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    get_constant_schedule_with_warmup
)

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed.utils import safe_get_full_grad


sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.data.data_utils import create_prompt_dataset
from utils.data.data_collator import DataCollator
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from utils.ds_utils import get_train_ds_config
from utils.module.lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, only_optimize_lora_parameters
from utils.model.model_utils import create_hf_model

# ------yen-------
# add flash attention
# from utils.flash_attention.llama_flash_att import replace_llama_attn_with_flash_attn
# from utils.flash_attention.bloom_flash_att import replace_bloom_attn_with_flash_attn

# replace_llama_attn_with_flash_attn()
# replace_bloom_attn_with_flash_attn()
# ------yen-------

from params import Method2Class, AllDatasetName


# TODO, check support for OPT and llama


def parse_args():
    def list_of_strings(arg):
        return arg.split(',')
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--data_path',
                        type=str,
                        default='Dahoas/rm-static',
                        help='Path to the training dataset, a single data path.')
    parser.add_argument('--dataset_name',
                        type=list_of_strings,
                        default='all',
                        help='Dataset to be used.')
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='/tmp/data_files/',
        help=
        'Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)'
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--max_prompt_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--max_ans_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs",
                        type=list_of_strings,
                        default=None,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--per-task-train-batch-sizes",
        type=list_of_strings,
        default=None,
        help="Per-task per-GPU micro-batch sizes (comma-separated, one per task).",
    )
    parser.add_argument(
        "--per-task-grad-accum-steps",
        type=list_of_strings,
        default=None,
        help="Per-task gradient accumulation steps (comma-separated). "
             "Use with per-task batch sizes to hold effective batch constant.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="A seed for reproducible training.")
    # local_rank usually indicates the index of the current process on the current node, while global_rank indicates the index of the current process among all processes.
    # When local_rank is -1, it means distributed training is not used. This value is usually set automatically by pytorch/deepspeed and does not need to be set by the user.
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--gradient_checkpointing',
                        action='store_true',
                        help='Enable HF gradient checkpointing for model.')
    # store_true means that if this argument is present in the command line, args.disable_dropout will be True; otherwise, it defaults to False
    parser.add_argument('--disable_dropout',
                        action='store_true',
                        help='Disable the dropout of the model.')
    # deepspeed features
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')
    
    ## Tensorboard logging
    parser.add_argument('--enable_tensorboard',
                        action='store_true',
                        help='Enable tensorboard logging')
    parser.add_argument('--tensorboard_path',
                        type=str,
                        default="step1_tensorboard")
    ## Print loss
    parser.add_argument('--print_loss',
                        action='store_true',
                        help='Prints loss at each step.')
    # CL_method
    parser.add_argument('--CL_method',
                default=None,
                help='continual learning method used')
    
    # Upcycle control: interval and explicit task names
    parser.add_argument('--upcycle-interval',
                        type=int,
                        default=4,
                        help='Number of tasks between upcycles (default 4). Set 1 to upcycle every task.')
    parser.add_argument('--upcycle-task-names',
                        type=str,
                        default='',
                        help='Comma-separated dataset names where upcycle should be forced (e.g. "ScienceQA,Py150").')
    parser.add_argument('--num-experts-per-task',
                        type=int,
                        default=8,
                        help='Number of experts to create per task (default: 8).')
    parser.add_argument('--num-activated-experts',
                        type=int,
                        default=2,
                        help='Number of experts activated per token (top-k routing, default: 2).')
    parser.add_argument('--task0-expansion-mode',
                        action='store_true',
                        help='Task 0 expansion mode: create 0~7 (frozen), then expand to 8~15 (trainable). Default: False (0~7 all trainable).')
    
    # ==========================================================================
    # Drop-Upcycling Arguments
    # ==========================================================================
    parser.add_argument('--ffn-init-ratio',
                        type=float,
                        default=0.5,
                        help='Ratio of FFN weights to re-initialize in Drop-Upcycling (0.0-1.0, default: 0.5).')
    parser.add_argument('--gate-init-method',
                        type=str,
                        default='torch_rand_002',
                        choices=['torch_rand', 'torch_rand_mean0', 'torch_normal_002', 'torch_normal_028', 'torch_rand_002'],
                        help='Method for initializing router gate weights (default: torch_rand_002).')
    
    # ==========================================================================
    # Branch-Train-Mix (BTM) Arguments
    # ==========================================================================
    parser.add_argument('--btm-source-paths',
                        type=str,
                        default='',
                        help='Comma-separated paths to source model checkpoints for BTM. If empty, uses base model (naive mode).')
    parser.add_argument('--btm-num-sources',
                        type=int,
                        default=4,
                        help='Number of source models for BTM (default: 4). Only used if btm-source-paths is not provided.')
    parser.add_argument('--btm-experts-per-source',
                        type=int,
                        default=2,
                        help='Number of experts per source model for BTM (default: 2).')
    
    # ==========================================================================
    # Expert Metrics Evaluation Arguments
    # ==========================================================================
    parser.add_argument('--enable-expert-metrics',
                        action='store_true',
                        help='Enable expert metrics evaluation (CKA, Flatness/Sharpness).')
    parser.add_argument('--cka-mode',
                        type=str,
                        default='full',
                        choices=['early', 'full'],
                        help='CKA computation mode: "early" for first N batches, "full" for entire dataset.')
    parser.add_argument('--cka-early-batches',
                        type=int,
                        default=10,
                        help='Number of batches for early CKA mode (default: 10).')
    parser.add_argument('--sharpness-epsilon',
                        type=float,
                        default=0.001,
                        help='Epsilon (perturbation radius) for epsilon-sharpness metric (default: 0.001).')
    parser.add_argument('--power-iterations',
                        type=int,
                        default=20,
                        help='Number of power iterations for Hessian eigenvalue estimation (default: 20).')
    parser.add_argument('--hutchinson-samples',
                        type=int,
                        default=10,
                        help='Number of samples for Hutchinson trace estimation (default: 10).')
    parser.add_argument('--metric-num-batches',
                        type=int,
                        default=5,
                        help='Number of batches for Hessian computation (default: 5).')
    parser.add_argument('--metric-checkpoint',
                        type=str,
                        default='after_train',
                        choices=['after_train', 'during_train', 'both'],
                        help='When to evaluate metrics: after_train, during_train, or both (default: after_train).')
    parser.add_argument('--metric-eval-interval',
                        type=int,
                        default=100,
                        help='Steps between metric evaluations during training (default: 100).')
    parser.add_argument('--metric-layer-idx',
                        type=int,
                        default=0,
                        help='Which MoE layer to evaluate (default: 0, first MoE layer).')
    parser.add_argument('--metric-all-layers',
                        action='store_true',
                        help='Evaluate all MoE layers instead of just one. Overrides --metric-layer-idx.')
    parser.add_argument('--metric-expert-scope',
                        type=str,
                        default='all',
                        choices=['current_task', 'all'],
                        help='Which experts to evaluate. "current_task": only current task\'s experts. "all": all experts across all tasks (default: all).')
    parser.add_argument('--metric-routing-mode',
                        type=str,
                        default='routed',
                        choices=['all', 'routed'],
                        help='Token selection for flatness. "routed": only tokens routed to each expert. "all": all tokens (default: routed). CKA always uses all tokens.')
    parser.add_argument('--cka-activation-source',
                        type=str,
                        default='output',
                        choices=['output', 'up_proj', 'gate_proj', 'gate_up'],
                        help='Which activation to use for CKA. "output": full expert output (default). "up_proj": up_proj output. "gate_proj": gate_proj after activation. "gate_up": concatenated gate+up.')
    parser.add_argument('--flatness-method',
                        type=str,
                        default='gradient_norm',
                        help='Primary flatness computation method (default: gradient_norm). Options: gradient_norm, landscape, fisher_trace, hessian.')
    parser.add_argument('--flatness-methods',
                        type=str,
                        default='gradient_norm,landscape',
                        help='Comma-separated list of flatness methods to compare (default: gradient_norm,landscape). Options: gradient_norm (~0.1s), landscape (~0.5s), fisher_trace (~1s), hessian (~5-10s). Use "all" for all methods.')
    parser.add_argument('--landscape-steps',
                        type=int,
                        default=10,
                        help='Number of steps for landscape flatness (default: 10). Only used when --flatness-method=landscape.')
    parser.add_argument('--landscape-multiplier',
                        type=float,
                        default=0.1,
                        help='Step size multiplier for landscape flatness (default: 0.1). Only used when --flatness-method=landscape.')
    parser.add_argument('--landscape-num-directions',
                        type=int,
                        default=3,
                        help='Number of random directions to average over (default: 3). Only used when --flatness-method=landscape.')
    
    # ==========================================================================
    # Router Initialization Configuration
    # ==========================================================================
    parser.add_argument('--router-init-method',
                        type=str,
                        default='random',
                        choices=['random', 'average', 'zero_bias', 'scaled_random', 'copy_with_noise'],
                        help='Router initialization method for new experts. Options: random (PyTorch default), average (use mean of old experts), zero_bias (negative bias to reduce initial routing), scaled_random (smaller random weights), copy_with_noise (copy average with small noise). Default: random.')
    
    # ==========================================================================
    # CKA Regularization Configuration (for CKA-guided training)
    # ==========================================================================
    parser.add_argument('--cka-regularization',
                        action='store_true',
                        help='Enable CKA regularization during training to preserve representations.')
    parser.add_argument('--cka-version',
                        type=str,
                        default='v5',
                        choices=['v5'],
                        help='Differentiable NAS-Guided Upcycling with Gumbel-Softmax masks. '
                             'Only the camera-ready v5 trainer is shipped in this release.')
    parser.add_argument('--lambda-cka',
                        type=float,
                        default=0.1,
                        help='CKA regularization strength (default: 0.1).')
    parser.add_argument('--cka-layers',
                        type=str,
                        default='deep',
                        help='Which layers to apply CKA regularization. Options: "all", "deep" (last 1/3), or comma-separated layer indices. Default: deep.')
    parser.add_argument('--cka-eval-interval',
                        type=int,
                        default=10,
                        help='Number of batches to cache for CKA baseline computation (default: 10).')
    parser.add_argument('--cka-compute-interval',
                        type=int,
                        default=50,
                        help='Compute CKA every N training steps (default: 50). Higher frequency for V2.')
    parser.add_argument('--cka-debug-interval',
                        type=int,
                        default=200,
                        help='Print detailed CKA debug info every N training steps (default: 200).')
    parser.add_argument('--cka-layer-weight-mode',
                        type=str,
                        default='last_layer_focused',
                        choices=['depth', 'last_layer_focused', 'empirical', 'adaptive', 'uniform'],
                        help='Layer weight mode for layerwise CKA. last_layer_focused (default): high weight on last layer based on empirical CKA findings. empirical: weights based on observed CKA drop. depth: deeper layers get higher weights. adaptive: based on measured CKA drop. uniform: equal weights.')
    parser.add_argument('--cka-task-weight-mode',
                        type=str,
                        default='type',
                        choices=['type', 'uniform'],
                        help='Task weight mode for layerwise CKA. type: generation tasks get higher weights for deep layers. uniform: equal weights. Default: type.')
    parser.add_argument('--cka-adapt-rate',
                        type=float,
                        default=0.5,
                        help='Adaptation rate for adaptive layer weights (default: 0.5).')
    parser.add_argument('--replay-buffer-size',
                        type=int,
                        default=100,
                        help='Maximum number of batches to store per task in replay buffer (default: 100).')
    parser.add_argument('--bilevel-weight-scale',
                        type=float,
                        default=0.1,
                        help='Scale factor for bilevel weight adjustment (default: 0.1).')
    parser.add_argument('--bilevel-base-weight',
                        type=float,
                        default=1.0,
                        help='Base weight for replay loss in bilevel optimization (default: 1.0).')
    parser.add_argument('--use-conflict-monitor',
                        action='store_true',
                        help='Enable conflict monitoring for dynamic expansion triggering.')
    parser.add_argument('--conflict-threshold',
                        type=float,
                        default=0.5,
                        help='Cosine similarity threshold for conflict detection (default: 0.5).')
    parser.add_argument('--conflict-patience',
                        type=int,
                        default=10,
                        help='Number of consecutive conflicts before triggering expansion signal (default: 10).')
    parser.add_argument('--start-task',
                        type=int,
                        default=0,
                        help='Task index to start training from (for resuming from checkpoint). Default: 0.')
    parser.add_argument('--resume-from',
                        type=str,
                        default=None,
                        help='Path to checkpoint directory to resume from (e.g., /path/to/output/3). '
                             'Loads MoE config, CKA baseline, and replay buffer.')
    parser.add_argument('--use-resumable-cka',
                        action='store_true',
                        help='Use resumable CKA implementation with automatic checkpointing.')
    parser.add_argument('--gradient-layers',
                        type=str,
                        default='router',
                        choices=['all', 'router', 'last'],
                        help='Which layers to use for gradient alignment computation. all: all parameters. router: only router layers. last: last 2 transformer layers. Default: router.')
    
    # Enhanced CKA parameters
    parser.add_argument('--use-adaptive-weight',
                        action='store_true',
                        default=True,
                        help='Use adaptive weight adjustment in enhanced CKA (default: True).')
    parser.add_argument('--max-weight-multiplier',
                        type=float,
                        default=1.5,
                        help='Maximum weight multiplier for adaptive CKA (default: 1.5). Lower than bilevel (3.0) to avoid over-regularization.')
    parser.add_argument('--task0-cka-weight',
                        type=float,
                        default=1.0,
                        help='CKA weight multiplier for Task 0 base alignment (default: 1.0). Set >1.0 to protect Task 0 knowledge.')
    parser.add_argument('--cka-warmup-ratio',
                        type=float,
                        default=0.0,
                        help='Ratio of training steps for CKA warmup (default: 0.0). Set 0.15 for 15%% warmup.')
    
    # Adaptive CKA parameters
    parser.add_argument('--layer-adjust-interval',
                        type=int,
                        default=100,
                        help='Steps between dynamic layer adjustment checks (default: 100).')
    parser.add_argument('--forgetting-threshold',
                        type=float,
                        default=0.02,
                        help='CKA drop threshold to trigger layer addition (default: 0.02 = 2%%).')
    
    # Loss-based early stopping parameters
    parser.add_argument('--early-stop-loss',
                        type=float,
                        default=0.01,
                        help='Loss threshold for early stopping (default: 0.01). Stop if mean loss < this.')
    parser.add_argument('--early-stop-min-epochs',
                        type=int,
                        default=2,
                        help='Minimum epochs before early stopping can trigger (default: 2).')

    # V3 Replay Training Loss parameters
    parser.add_argument('--replay-weight',
                        type=float,
                        default=0.5,
                        help='Weight for replay training loss (default: 0.5). Only used in CKA V3.')
    parser.add_argument('--replay-freq',
                        type=int,
                        default=1,
                        help='Compute replay loss every N steps (default: 1). Increase to reduce overhead.')
    parser.add_argument('--use-weighted-replay',
                        action='store_true',
                        default=True,
                        help='Use weighted sampling for replay (older tasks get higher weight). Default: True.')
    parser.add_argument('--freeze-lm-head',
                        action='store_true',
                        default=False,
                        help='Freeze lm_head to prevent output layer forgetting. Only used in CKA V3.')
    parser.add_argument('--use-layerwise-penalty',
                        action='store_true',
                        default=True,
                        help='Use layer-wise adaptive CKA penalty (V3). Deeper layers get higher weights.')
    parser.add_argument('--no-layerwise-penalty',
                        dest='use_layerwise_penalty',
                        action='store_false',
                        help='Disable layer-wise penalty, use uniform weights for all layers.')

    # === V4: NAS-Guided Upcycling Arguments ===
    parser.add_argument('--nas-threshold-high',
                        type=float,
                        default=0.6,
                        help='V4: Sensitivity threshold for EXPAND decision. Experts above this are critical. Default: 0.6')
    parser.add_argument('--nas-threshold-low',
                        type=float,
                        default=0.2,
                        help='V4: Sensitivity threshold for RECYCLE decision. Experts below this are redundant. Default: 0.2')
    parser.add_argument('--nas-probe-batches',
                        type=int,
                        default=20,
                        help='V4: Number of replay batches to use for sensitivity probing. Default: 20')
    parser.add_argument('--no-nas-probe',
                        dest='nas_probe_enabled',
                        action='store_false',
                        default=True,
                        help='V4: Disable NAS probing, use standard upcycling.')

    # === V5: Differentiable NAS Arguments ===
    parser.add_argument('--nas-temperature-init',
                        type=float,
                        default=5.0,
                        help='V5: Initial Gumbel temperature for soft mask sampling. Default: 5.0')
    parser.add_argument('--nas-temperature-final',
                        type=float,
                        default=0.1,
                        help='V5: Final Gumbel temperature after annealing. Default: 0.1')
    parser.add_argument('--nas-decay-rate',
                        type=float,
                        default=0.99,
                        help='V5: Temperature decay rate per mask update. Default: 0.99')
    parser.add_argument('--mask-update-interval',
                        type=int,
                        default=10,
                        help='V5: Update mask parameters every N training steps. Default: 10')
    parser.add_argument('--sparsity-weight',
                        type=float,
                        default=0.1,
                        help='V5: Sparsity penalty coefficient (encourages RECYCLE). Default: 0.1')
    parser.add_argument('--mask-lr',
                        type=float,
                        default=0.01,
                        help='V5: Learning rate for mask optimizer. Default: 0.01')
    parser.add_argument('--nas-layers',
                        type=str,
                        default='all',
                        help='V5: Which layers to apply NAS. Options: "all", "last_N" (e.g., last_4), or comma-separated indices. Default: all')
    parser.add_argument('--deep-layer-expand-bias',
                        type=float,
                        default=0.0,
                        help='V5: Extra expand bias for deep layers (0-2 range). Higher = more expansion in deep layers. Default: 0.0')
    parser.add_argument('--deep-layer-indices',
                        type=str,
                        default=None,
                        help='V5: Comma-separated layer indices considered "deep" (e.g., "13,14,15"). Default: last 3 NAS layers')
    parser.add_argument('--no-sensitivity-init',
                        dest='use_sensitivity_init',
                        action='store_false',
                        default=True,
                        help='V5: Disable using V4 sensitivity scores to initialize masks.')
    parser.add_argument('--bilevel-log-interval',
                        type=int,
                        default=10,
                        help='V5: Log bilevel optimization state every N mask updates. Default: 10')
    parser.add_argument('--bilevel-log-detailed',
                        action='store_true',
                        default=True,
                        help='V5: Enable detailed per-layer mask logging. Default: True')
    parser.add_argument('--no-bilevel-log-detailed',
                        dest='bilevel_log_detailed',
                        action='store_false',
                        help='V5: Disable detailed per-layer mask logging.')
    
    # V5 Balanced Loss Configuration (fix for all-RECYCLE collapse)
    parser.add_argument('--knowledge-gain-weight',
                        type=float,
                        default=0.3,
                        help='V5: Weight for knowledge gain signal (encourages EXPAND when new experts are better). Default: 0.3')
    parser.add_argument('--target-expand-rate',
                        type=float,
                        default=0.3,
                        help='V5: Target expansion rate (fraction of experts to expand). Default: 0.3 (30%%)')
    parser.add_argument('--balance-weight',
                        type=float,
                        default=0.5,
                        help='V5: Penalty weight for under-expansion (deviation from target). Default: 0.5')
    parser.add_argument('--use-adaptive-weights',
                        action='store_true',
                        default=True,
                        help='V5: Enable adaptive weight adjustment based on training signals. Default: True')
    parser.add_argument('--no-adaptive-weights',
                        dest='use_adaptive_weights',
                        action='store_false',
                        help='V5: Disable adaptive weight adjustment.')
    parser.add_argument('--adaptive-window-size',
                        type=int,
                        default=50,
                        help='V5: Window size for computing adaptive weight signals. Default: 50')
    
    # Task Protection (for reducing catastrophic forgetting)
    parser.add_argument('--use-task-protection',
                        action='store_true',
                        default=True,
                        help='V5: Enable progressive task protection. Default: True')
    parser.add_argument('--no-task-protection',
                        dest='use_task_protection',
                        action='store_false',
                        help='V5: Disable task protection.')
    parser.add_argument('--task-protection-factor',
                        type=float,
                        default=1.5,
                        help='V5: Replay weight multiplier for oldest task (linear decay). Default: 1.5')
    parser.add_argument('--min-replay-per-task',
                        type=int,
                        default=20,
                        help='V5: Minimum replay samples to maintain per task. Default: 20')

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()


    return args


def main():
    args = parse_args()

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()

    ds_config = get_train_ds_config(offload=args.offload,
                                    stage=args.zero_stage,
                                    enable_tensorboard=args.enable_tensorboard,
                                    tb_path=args.tensorboard_path,
                                    tb_name="v2_sft")
    # set batch size
    ds_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config[
        'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size(
        ) * args.gradient_accumulation_steps

    # If passed along, set the training seed now.
    set_random_seed(args.seed)
    # Barrier to make sure all process are ready to train
    torch.distributed.barrier()

    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
    # default the LLM is decoder only model, so padding side is left
    assert tokenizer.padding_side == 'left'
    assert tokenizer.truncation_side == "left"

    model = create_hf_model(AutoModelForCausalLM,
                            args.model_name_or_path,
                            tokenizer,
                            ds_config=ds_config,
                            disable_dropout=args.disable_dropout
                            )
    
    train_task_list = {}
    eval_task_list = {}
    test_task_list = {}
    train_dataset_list = {}


    if args.dataset_name[0] == "all":
        Datasets = AllDatasetName
    else:
        Datasets = args.dataset_name
    for dataset in Datasets:
        dataset_path = os.path.join(args.data_path,dataset)
        # Prepare the data
        train_dataset, eval_dataset, test_dataset = create_prompt_dataset(
            args.local_rank,
            dataset_path,
            args.data_output_path,
            args.seed
        )

        # DataLoaders creation:
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_dataset)
            eval_sampler = SequentialSampler(eval_dataset)
            test_sampler = SequentialSampler(test_dataset)

        else:
            train_sampler = DistributedSampler(train_dataset)
            eval_sampler = DistributedSampler(eval_dataset)
            test_sampler = DistributedSampler(test_dataset)

        data_collator = DataCollator(
            tokenizer,
            padding="longest",
            max_prompt_len=args.max_prompt_len,
            max_ans_len=args.max_ans_len,
            pad_to_multiple_of=8,
            inference=False
        )
        inf_data_collator = DataCollator(
            tokenizer,
            model=model,
            padding="longest",
            max_prompt_len=args.max_prompt_len,
            max_ans_len=args.max_ans_len,
            pad_to_multiple_of=8,
            inference=True
        )
                

        train_dataloader = DataLoader(train_dataset,
                                    collate_fn=data_collator,
                                    sampler=train_sampler,
                                    batch_size=args.per_device_train_batch_size,
                                    num_workers=4,
                                    pin_memory=True)
        eval_dataloader = DataLoader(eval_dataset,
                                    collate_fn=data_collator,
                                    sampler=eval_sampler,
                                    batch_size=args.per_device_eval_batch_size,
                                    num_workers=4,
                                    pin_memory=True)
        test_dataloader = DataLoader(test_dataset,
                            collate_fn=inf_data_collator,
                            sampler=test_sampler,
                            batch_size=args.per_device_eval_batch_size,
                            num_workers=4,
                            pin_memory=True)
        train_task_list[dataset] = train_dataloader
        eval_task_list[dataset] = eval_dataloader
        test_task_list[dataset] = test_dataloader
        train_dataset_list[dataset] = train_dataset

    args.train_dataset_list = train_dataset_list
    args.data_collator = data_collator


    def evaluation(model, eval_dataloader):
        model.eval()
        losses = 0
        for step, batch in enumerate(eval_dataloader):
            # implementation, batch = {k: v.to(device) for k, v in batch.items()}
            del batch['sources']
            batch = to_device(batch, device)
            with torch.no_grad():
                # TODO, check output
                outputs = model(**batch)

            loss = outputs.loss
            losses += loss.float()
        losses = losses / (step + 1)
        try:
            perplexity = torch.exp(losses)
        except OverflowError:
            perplexity = float("inf")
        try:
            perplexity = get_all_reduce_mean(perplexity).item()
        except:
            pass
        return perplexity

    def get_optimizer(model):
        # Split weights in two groups, one with weight decay and the other not.
        optimizer_grouped_parameters = get_optimizer_grouped_parameters(
            model, args.weight_decay)

        AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
        optimizer = AdamOptimizer(optimizer_grouped_parameters,
                                lr=args.learning_rate,
                                betas=(0.9, 0.95))
        
        total_train_dataloader_len = sum(len(train_task_list[task]) for task in list(train_task_list.keys()))
        num_update_steps_per_epoch = math.ceil(
            total_train_dataloader_len / args.gradient_accumulation_steps)
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps
        )
        
        return optimizer, lr_scheduler
    
    optimizer, lr_scheduler = get_optimizer(model)
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Train!
    print_rank_0("***** Running training *****", args.global_rank)
    # print_rank_0(
    #     f"***** Evaluating perplexity, Epoch {0}/{args.num_train_epochs} *****",
    #     args.global_rank)
    # perplexity = evaluation(model, eval_dataloader)
    # print_rank_0(f"ppl: {perplexity}", args.global_rank)

    # Initialize the global progress bar

    if args.CL_method in Method2Class.keys():
        # Open-source release: only the V5 (Differentiable NAS + Gumbel-Softmax)
        # CKA-guided trainer is exposed. Earlier ablation variants (V2/V3/V4)
        # and the orthogonal V7 merge exploration live in the full research
        # repository and are not part of the camera-ready reproduction path.
        if args.CL_method == "upcycle" and getattr(args, 'cka_regularization', False):
            cka_version = getattr(args, 'cka_version', 'v5')
            if cka_version != 'v5':
                raise ValueError(
                    f"Only --cka-version v5 is supported in this release; got '{cka_version}'."
                )
            from model.Dynamic_network.upcycling_cka_v5 import create_cka_upcycle_v5
            print_rank_0(
                f"[Main] CKA V5 (Differentiable NAS + Gumbel-Softmax) | "
                f"temperature={args.nas_temperature_init}→{args.nas_temperature_final}, "
                f"decay={args.nas_decay_rate}, mask_interval={args.mask_update_interval}, "
                f"sparsity_weight={args.sparsity_weight}, nas_layers={args.nas_layers}",
                args.global_rank,
            )
            CL_Trainer = create_cka_upcycle_v5(
                model, tokenizer, optimizer, train_task_list, eval_task_list,
                test_task_list, args,
            )
        else:
            CL_Trainer = Method2Class[args.CL_method](
                model, tokenizer, optimizer, train_task_list, eval_task_list,
                test_task_list, args,
            )
        CL_Trainer.train_continual()


if __name__ == "__main__":
    main()
