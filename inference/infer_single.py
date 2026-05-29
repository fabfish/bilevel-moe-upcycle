"""
    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
"""

# !/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import os
import math
import sys
from tqdm import tqdm
import pandas as pd

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import deepspeed
import json

from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoModelForCausalLM,
)

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.data.data_collator import DataCollator
from utils.data.data_utils import create_prompt_dataset
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, \
    get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from utils.ds_utils import get_train_ds_config
from utils.model.model_utils import create_hf_model
from training.params import Method2Class, AllDatasetName

# Use upcycling_refactored.py (not upcycling.py) to match training code
from model.Dynamic_network.upcycling_refactored import convert_upcycle_model, convert_upcycle_model_variable

# dist.init_process_group(backend='nccl')

# Module version info for debugging
_UPCYCLE_MODULE_INFO = {
    'module': 'upcycling_refactored.py',
    'import_path': 'model.Dynamic_network.upcycling_refactored'
}


def parse_args():
    def list_of_strings(arg):
        return arg.split(',')
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--data_path',
                        type=str,
                        default='Dahoas/rm-static',
                        help='Path to the training dataset. A single data path.')
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
        "--inference_model_path",
        type=str,
        help=
        "Path to inference model.",
        required=True,
    )
    parser.add_argument(
        "--max_prompt_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    # inference params
    parser.add_argument(
        "--max_ans_len",
        type=int,
        default=256,
        help="The maximum answer length.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Generate temperature params.",
    )

    parser.add_argument(
        "--inference_batch",
        type=int,
        default=4,
        help="Inference batch size.",
    )
    # TODO, add other inference params
    parser.add_argument(
        "--inference_tasks",
        type=list_of_strings,
        default='all',
        help='Datasets to be used.'
    )
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="A seed for reproducible training.")

    # local_rank 一般表示当前进程在当前节点的编号，global_rank 表示当前进程在所有进程中的编号
    # local_rank 为 -1 时，表示不使用分布式训练。这个值一般由 pytorch/deepspeed 自动设置，用户不用管
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")

    # added by wangxiao
    parser.add_argument('--inference_output_path',
                        type=str,
                        default=None,
                        help="Where to store inference results.")
    parser.add_argument('--CL_method',
                        default=None,
                        help='continual learning method used')
    
    # Upcycle parameters
    parser.add_argument('--num_experts_per_task',
                        type=int,
                        default=8,
                        help="Number of experts per task for MoE upcycling.")
    parser.add_argument('--num_activated_experts',
                        type=int,
                        default=2,
                        help="Number of activated experts per token.")
    parser.add_argument('--upcycle-interval',
                        type=int,
                        default=4,
                        help="Number of tasks between upcycles when converting model for inference (default 4).")
    parser.add_argument('--upcycle-task-names',
                        type=str,
                        default='',
                        help='Comma-separated dataset names where upcycle should be forced during inference.')
    parser.add_argument('--router_init_method',
                        type=str,
                        default='random',
                        choices=['random', 'average', 'zero_bias', 'scaled_random', 'copy_with_noise'],
                        help='Router initialization method for new experts.')
    parser.add_argument('--metric_routing_mode',
                        type=str,
                        default='routed',
                        choices=['all', 'routed'],
                        help='Whether to use routing-aware metrics (routed) or all tokens (all).')
    
    # Parallel execution support
    parser.add_argument('--specific-round',
                        type=int,
                        default=None,
                        help='Run inference for a specific round only (for parallel execution). '
                             'If not set, runs all rounds sequentially.')

    # Profiling helper: limit number of inference steps (batches)
    parser.add_argument('--max_steps',
                        type=int,
                        default=None,
                        help='Optional max number of inference steps (batches) to run. '
                             'Useful for quick profiling; by default runs all steps.')

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    set_random_seed(args.seed)
    device = torch.device("cuda")


    def prediction(model, infer_dataloader):
        predicted_sequences = []
        sources_sequences = []
        ground_truths = []
        model.eval()
        for step, batch in enumerate(infer_dataloader):
            # TODO, add prompts, choosen, rejected
            # implementation, batch = {k: v.to(device) for k, v in batch.items()}
            sources_sequences += batch['sources']
            ground_truths += batch['gts']
            del batch['sources']
            del batch['gts']
            batch = to_device(batch, device)
            prompt_len = batch['input_ids'].shape[1]
            # update progress bar
            progress_bar.update(1)
            description = f"Step {step}"
            progress_bar.set_description(description, refresh=False)
            with torch.no_grad():
                # TODO, add more inference params
                # backbone config
                # generate_ids = model.generate(batch['input_ids'], max_new_tokens=args.max_ans_len,
                #                               pad_token_id=tokenizer.eos_token_id, attention_mask = batch['attention_mask'], temperature=0.7, do_sample=True, repetition_penalty=2.0 )
                # sft config
                generate_ids = model.generate(input_ids=batch['input_ids'],
                                              attention_mask=batch['attention_mask'],
                                              max_new_tokens=args.max_ans_len,
                                              bos_token_id=tokenizer.bos_token_id,
                                              eos_token_id=tokenizer.eos_token_id,
                                              pad_token_id=tokenizer.unk_token_id,
                                              temperature=args.temperature,
                                              do_sample=True,
                                              num_return_sequences=1,
                                              use_cache=True
                                              )
            sequences = tokenizer.batch_decode(generate_ids[:, prompt_len:], skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False)
            predicted_sequences += sequences

            # Optional early break for profiling
            if args.max_steps is not None and (step + 1) >= args.max_steps:
                break
        return sources_sequences, predicted_sequences, ground_truths

    def get_result_file_path(round: int, i_task: int, task: str):
        """Get the path for a result file."""
        return os.path.join(args.inference_output_path, f"results-{round}-{i_task}-{task}.json")
    
    def check_result_exists(round: int, i_task: int, task: str):
        """Check if a result file already exists (for resume support)."""
        result_path = get_result_file_path(round, i_task, task)
        return os.path.exists(result_path)
    
    def check_round_complete(round: int):
        """Check if all results for a round already exist."""
        for i_task in range(round + 1):
            task = inference_tasks[i_task]
            if not check_result_exists(round, i_task, task):
                return False
        return True

    def save_inference_results(evaluation_result: dict, sources_sequences: list, predicted_sequences: list,
                                ground_truths: list, round: int, i_task: int, task: str):
        # save as a json file
        df = {"eval": evaluation_result, 'prompts': sources_sequences, 'results': predicted_sequences,
                'labels': ground_truths}
        if not os.path.exists(args.inference_output_path):
            os.makedirs(args.inference_output_path)
        with open(args.inference_output_path + "/results-" + str(round) + "-" + str(i_task) + "-" + task + ".json", "w+", encoding='utf-8') as file:
            json.dump(df, file, ensure_ascii=False)


    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
    # default the LLM is decoder only model, so padding side is left
    assert tokenizer.padding_side == 'left'
    assert tokenizer.truncation_side == "left"

    # set evaluation batch size
    # only support bs = 1, cause right padding training logic
    # TODO, modify left pad for training and inference
    inference_tasks = args.inference_tasks 
    task_num = len(inference_tasks)
    
    # Determine which rounds to run
    if args.specific_round is not None:
        # Run only the specified round (for parallel execution)
        if args.specific_round < 0 or args.specific_round >= task_num:
            print_rank_0(f"Error: --specific-round {args.specific_round} is out of range [0, {task_num-1}]", args.local_rank)
            return
        rounds_to_run = [args.specific_round]
        print_rank_0(f"***** Running inference for specific round {args.specific_round} only *****", args.local_rank)
    else:
        # Run all rounds sequentially
        rounds_to_run = list(range(task_num))
    
    for round in rounds_to_run:   # load models and adapters of a new round in continual learning
        # Skip if all results for this round already exist (resume support)
        if check_round_complete(round):
            print_rank_0(f"***** Skipping round {round} - all results already exist *****", args.local_rank)
            continue
        
        inference_model_path = os.path.join(args.inference_model_path, str(round))
        print_rank_0("Inference Model Path: " + inference_model_path, args.local_rank)

        model = create_hf_model(AutoModelForCausalLM,
                                args.model_name_or_path,
                                tokenizer,
                                ds_config=None,
                                )
        
        # TODO: add adapters
        if args.CL_method in ("upcycle", "drop_upcycle", "btm"):
            # MoE model loading (upcycle / drop_upcycle / btm share same scientific_experts + router structure)
            import traceback
            method_tag = "DropUpcycle" if args.CL_method == "drop_upcycle" else ("BTM" if args.CL_method == "btm" else "Upcycle")
            print_rank_0(f"[Infer][{method_tag}] Loading MoE checkpoint", args.local_rank)
            args.num_experts_per_task = getattr(args, "num_experts_per_task", 8)
            args.num_activated_experts = getattr(args, "num_activated_experts", 2)
            print_rank_0(f"[Infer][{method_tag}] Config: num_experts_per_task={args.num_experts_per_task}, num_activated_experts={args.num_activated_experts}", args.local_rank)

            # Determine if we should convert/load MoE this round
            if args.CL_method in ("drop_upcycle", "btm"):
                do_upcycle = True  # always MoE per round
            else:
                current_task_name = inference_tasks[round] if round < len(inference_tasks) else None
                raw_names = getattr(args, "upcycle_task_names", "")
                upcycle_names = {n.strip() for n in raw_names.split(",") if n.strip()} if isinstance(raw_names, str) else set(raw_names or [])
                do_upcycle = bool(args.upcycle_interval and (round % max(1, args.upcycle_interval) == 0)) or (current_task_name in upcycle_names)
            print_rank_0(f"[Infer][{method_tag}] round={round} do_upcycle={do_upcycle}", args.local_rank)

            ckpt_path = os.path.join(inference_model_path, "pytorch_model.bin")
            print_rank_0(f"[Infer][Upcycle] Loading checkpoint from: {ckpt_path}", args.local_rank)
            
            if os.path.isfile(ckpt_path):
                state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                print_rank_0(f"[Infer][Upcycle] Checkpoint loaded: {len(state_dict)} parameters", args.local_rank)

                # Find router output dimension from checkpoint
                router_out_dim = None
                expert_count_by_layer = {}
                for k, v in state_dict.items():
                    if k.endswith(".mlp.router.classifier.weight"):
                        try:
                            router_out_dim = v.shape[0]
                            # Extract layer index
                            layer_idx = int(k.split('.')[2])
                            expert_count_by_layer[layer_idx] = v.shape[0]
                        except Exception:
                            continue
                    # Count scientific experts
                    if ".mlp.scientific_experts." in k and ".gate_proj.weight" in k:
                        try:
                            parts = k.split(".")
                            layer_idx = int(parts[2])
                            expert_idx = int(parts[5])
                            if layer_idx not in expert_count_by_layer:
                                expert_count_by_layer[layer_idx] = 0
                            expert_count_by_layer[layer_idx] = max(expert_count_by_layer[layer_idx], expert_idx + 1)
                        except Exception:
                            continue

                if router_out_dim is not None:
                    per_task = int(getattr(args, "num_experts_per_task", 8))
                    
                    # Check if we have variable expert counts per layer (V4/V5 selective expansion)
                    unique_counts = set(expert_count_by_layer.values())
                    has_variable_experts = len(unique_counts) > 1 or (len(unique_counts) == 1 and list(unique_counts)[0] != per_task)
                    
                    print_rank_0(f"[Infer][Upcycle] Checkpoint analysis:", args.local_rank)
                    print_rank_0(f"[Infer][Upcycle]   Expert counts by layer: {expert_count_by_layer}", args.local_rank)
                    print_rank_0(f"[Infer][Upcycle]   Has variable experts: {has_variable_experts}", args.local_rank)
                    
                    try:
                        if has_variable_experts and len(expert_count_by_layer) > 0:
                            # V4/V5 selective expansion: use variable expert counts
                            print_rank_0(f"[Infer][Upcycle] Using VARIABLE expert counts (V4/V5 mode)", args.local_rank)
                            model = convert_upcycle_model_variable(model, args, expert_count_by_layer)
                        else:
                            # Standard upcycling: uniform expert counts
                            num_tasks_needed = math.ceil(router_out_dim / per_task)
                            print_rank_0(f"[Infer][Upcycle] Using UNIFORM expert counts: num_tasks={num_tasks_needed}", args.local_rank)
                            model = convert_upcycle_model(model, args, num_tasks=num_tasks_needed, incremental=False)
                    except Exception as e:
                        print_rank_0(f"[Infer][Upcycle] ERROR during model conversion: {str(e)}", args.local_rank)
                        print_rank_0(f"[Infer][Upcycle] Traceback:\n{traceback.format_exc()}", args.local_rank)
                        raise
                else:
                    print_rank_0(f"[Infer][Upcycle] No router found in checkpoint, using incremental mode", args.local_rank)
                    if do_upcycle:
                        print_rank_0(f"[Infer][Upcycle] Conversion params: num_tasks=1, incremental=True", args.local_rank)
                        try:
                            model = convert_upcycle_model(model, args, num_tasks=1, incremental=True)
                        except Exception as e:
                            print_rank_0(f"[Infer][Upcycle] ERROR during model conversion: {str(e)}", args.local_rank)
                            print_rank_0(f"[Infer][Upcycle] Traceback:\n{traceback.format_exc()}", args.local_rank)
                            raise

                # Load checkpoint weights with enhanced logging
                print_rank_0(f"[Infer][Upcycle] Loading state dict into model...", args.local_rank)
                try:
                    model.load_state_dict(state_dict, strict=True)
                    print_rank_0(f"[Infer][Upcycle] ✓ Loaded checkpoint with strict=True", args.local_rank)
                except Exception as e:
                    print_rank_0(f"[Infer][Upcycle] strict=True failed: {e}", args.local_rank)
                    print_rank_0(f"[Infer][Upcycle] Retrying with strict=False...", args.local_rank)
                    try:
                        model.load_state_dict(state_dict, strict=False)
                        print_rank_0(f"[Infer][Upcycle] ✓ Loaded checkpoint with strict=False", args.local_rank)
                    except Exception as e2:
                        print_rank_0(f"[Infer][Upcycle] ERROR: Failed to load checkpoint: {str(e2)}", args.local_rank)
                        print_rank_0(f"[Infer][Upcycle] Traceback:\n{traceback.format_exc()}", args.local_rank)
                        raise
                
                # Log model structure after loading
                sample_mlp = model.model.layers[0].mlp
                if hasattr(sample_mlp, 'scientific_experts'):
                    num_experts_loaded = len(sample_mlp.scientific_experts)
                    router_dim = sample_mlp.router.classifier.out_features if hasattr(sample_mlp, 'router') else 0
                    print_rank_0(f"[Infer][Upcycle] Model structure: {num_experts_loaded} experts, router_dim={router_dim}", args.local_rank)
                
                del state_dict
            else:
                print_rank_0(f"[Infer][Upcycle] WARNING: checkpoint not found at {ckpt_path}", args.local_rank)
                if do_upcycle:
                    print_rank_0(f"[Infer][Upcycle] Converting model with num_tasks=1, incremental=True", args.local_rank)
                    try:
                        model = convert_upcycle_model(model, args, num_tasks=1, incremental=True)
                    except Exception as e:
                        print_rank_0(f"[Infer][Upcycle] ERROR during model conversion: {str(e)}", args.local_rank)
                        print_rank_0(f"[Infer][Upcycle] Traceback:\n{traceback.format_exc()}", args.local_rank)
                        raise

        if args.CL_method not in ("lora", "O-LoRA", "LFPT5", "upcycle", "drop_upcycle", "btm"):
            inference_model = torch.load(os.path.join(inference_model_path, "pytorch_model.bin"), weights_only=False)
            for name, param in model.named_parameters():
                param.data.copy_(inference_model[name])
            del inference_model

        model.to(device)

        for inference_task_id in range(round+1):    # evaluation for previous tasks in a single round
            inference_task = inference_tasks[inference_task_id]
            
            # Skip if result already exists (resume support)
            if check_result_exists(round, inference_task_id, inference_task):
                print_rank_0(f"***** Skipping round={round}, task={inference_task_id} ({inference_task}) - result already exists *****", args.local_rank)
                continue
            
            dataset_path = os.path.join(args.data_path, inference_task)
            # Prepare the data
            _, _, infer_dataset = create_prompt_dataset(
                args.local_rank,
                dataset_path,
                args.data_output_path,
                args.seed,
                distributed=False
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
            infer_sampler = SequentialSampler(infer_dataset)
            infer_dataloader = DataLoader(infer_dataset,
                                          collate_fn=inf_data_collator,
                                          sampler=infer_sampler,
                                          batch_size=args.inference_batch)
            progress_bar = tqdm(total=len(infer_dataloader), leave=True)

            # Inference !
            print_rank_0("***** Start inference *****", args.local_rank)
            sources_sequences, predicted_sequences, ground_truths = prediction(model, infer_dataloader)
            
            # Get Accuracy/ROUGE/BLEU/...
            # The evaluation result is stored in a dictionary. e.g. {"accuracy": .., "rouge-L": ..}
            # TRACE task scorers are imported lazily: they are part of the TRACE
            # benchmark (not this work) and are only needed at scoring time.
            from evaluations import (
                eval_ScienceQA, eval_MeetingBank, eval_PapyrusF, eval_CStance,
                eval_Py150, eval_FOMC, eval_NumGLUE_cm, eval_NumGLUE_ds, eval_20Minuten,
            )
            if inference_task == "ScienceQA":
                evaluation_result = eval_ScienceQA.eval(predicted_sequences, ground_truths)
            elif inference_task == "MeetingBank":
                evaluation_result = eval_MeetingBank.eval(predicted_sequences, ground_truths)
            elif inference_task == "C-STANCE":
                evaluation_result = eval_CStance.eval(predicted_sequences, ground_truths)
            elif inference_task == "Papyrus-f":
                evaluation_result = eval_PapyrusF.eval(predicted_sequences, ground_truths)
            elif inference_task == "Py150":
                evaluation_result = eval_Py150.eval(predicted_sequences, ground_truths)
            elif inference_task == "FOMC":
                evaluation_result = eval_FOMC.eval(predicted_sequences, ground_truths)
            elif inference_task == "NumGLUE-cm":
                evaluation_result = eval_NumGLUE_cm.eval(predicted_sequences, ground_truths)
            elif inference_task == "NumGLUE-ds":
                evaluation_result = eval_NumGLUE_ds.eval(predicted_sequences, ground_truths)
            elif inference_task == "20Minuten":
                evaluation_result = eval_20Minuten.eval(sources_sequences, predicted_sequences, ground_truths)
            else:
                evaluation_result = {}

            # if args.global_rank <= 0:  # only one process is running
            print("***** Saving inference results *****")
            save_inference_results(evaluation_result, sources_sequences, predicted_sequences, ground_truths, round, inference_task_id, inference_task)

if __name__ == "__main__":
    main()
