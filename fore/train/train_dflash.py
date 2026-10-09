#!/usr/bin/env python3
# coding=utf-8
"""DFlash Training Script."""

import argparse
import logging
import math
import os
import shutil
import time
import warnings
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from accelerate.utils import set_seed
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from datasets import load_dataset
from fore.train.args import TrackerArgs
from fore.train.core.dflash import OnlineDFlashModel, variable_prefix_probs
from fore.train.data import build_eagle3_dataset, prepare_dp_dataloaders
from fore.train.distributed import destroy_distributed, get_dp_group, init_distributed
from fore.train.modeling.draft.dflash import (
    DFlashDraftModel,
)
from fore.train.modeling.target.dflash_target_model import (
    DFlashTargetModel,
    get_dflash_target_model,
)
from fore.train.modeling.target.target_utils import TargetEmbeddingsAndHead
from fore.train.optimizer import BF16Optimizer
from fore.train.tracker import create_tracker
from fore.train.utils import get_last_checkpoint, print_on_rank0, print_with_rank


def parse_args():
    parser = argparse.ArgumentParser(description="Train DFlash Draft Model")

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--target-model-path", type=str, required=True)
    model_group.add_argument(
        "--target-model-backend",
        type=str,
        default="hf",
        choices=["hf"],
        help="Backend for target model: 'hf' (local)",
    )
    model_group.add_argument("--draft-config-path", type=str, default=None)
    # FORE-GRAFT: fine-tune an existing drafter (e.g. the released Qwen3.5-9B-DFlash) under a
    # new recipe: config + weights come from PATH, optimizer/scheduler/step stay fresh.
    # --resume wins if output-dir has a checkpoint.
    model_group.add_argument("--init-from", type=str, required=True)
    # FORE-VARIABLE-PREFIX: variable-length true-prefix forcing. --variable-prefix
    # loads the length distribution used for the FoRE drafters.
    model_group.add_argument("--variable-prefix", action="store_true")
    model_group.add_argument("--block-size", type=int, default=16)
    model_group.add_argument(
        "--mask-token-id",
        type=int,
        default=None,
        help="MASK token ID. If not provided, auto-detect from tokenizer.",
    )
    model_group.add_argument(
        "--attention-backend",
        type=str,
        default="flex_attention",
        choices=["eager", "sdpa", "flex_attention"],
        help="Attention backend for draft model.",
    )
    model_group.add_argument(
        "--trust-remote-code", action="store_true", help="Trust remote code"
    )
    model_group.add_argument(
        "--num-anchors",
        type=int,
        default=512,
        help="Number of anchor positions per sequence",
    )
    model_group.add_argument(
        "--loss-type",
        type=str,
        default="dpace",
        choices=[
            "dpace",
        ],
        help=("Loss variant. Use dpace for Dynamic Position-Aware Cross-Entropy."),
    )
    model_group.add_argument(
        "--dpace-alpha",
        type=float,
        default=0.5,
        help="Smoothing alpha for D-PACE position weights.",
    )
    model_group.add_argument(
        "--embedding-key",
        type=str,
        default=None,
        help="Embedding weight key in the target model. "
        "Default: 'model.embed_tokens.weight' for standard models, "
        "'model.language_model.embed_tokens.weight' for multimodal models such as Qwen3.5.",
    )
    model_group.add_argument(
        "--lm-head-key",
        type=str,
        default=None,
        help="LM head weight key in the target model. Default: 'lm_head.weight'.",
    )

    dataset_group = parser.add_argument_group("dataset")
    dataset_group.add_argument("--train-data-path", type=str, required=True)
    dataset_group.add_argument("--chat-template", type=str, default="qwen3-nosys")
    dataset_group.add_argument("--is-preformatted", action="store_true")
    dataset_group.add_argument("--dataloader-num-workers", type=int, default=8)
    dataset_group.add_argument(
        "--build-dataset-num-proc",
        type=int,
        default=int(os.environ.get("SPECFORGE_DATA_NUM_PROC", 8)),
    )

    training_group = parser.add_argument_group("training")
    training_group.add_argument("--num-epochs", type=int, default=6)
    training_group.add_argument("--batch-size", type=int, default=1)
    training_group.add_argument("--learning-rate", type=float, default=6e-4)
    training_group.add_argument("--max-length", type=int, default=3072)
    training_group.add_argument("--warmup-ratio", type=float, default=0.04)
    training_group.add_argument("--max-grad-norm", type=float, default=1.0)
    training_group.add_argument("--weight-decay", type=float, default=0.0)
    training_group.add_argument("--accumulation-steps", type=int, default=1)
    training_group.add_argument("--seed", type=int, default=42)
    training_group.add_argument("--resume", action="store_true")

    output_group = parser.add_argument_group("output")
    output_group.add_argument("--output-dir", type=str, required=True)
    output_group.add_argument("--cache-dir", type=str, default="./cache")
    output_group.add_argument("--log-interval", type=int, default=50)
    output_group.add_argument("--save-interval", type=int, default=1000)
    output_group.add_argument(
        "--save-total-limit",
        type=int,
        default=3,
        help="keep only the N most recent checkpoints (0 = keep all)",
    )

    optimization_group = parser.add_argument_group("optimization")
    optimization_group.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="The size of the tensor parallel for the target model",
    )

    tracker_group = parser.add_argument_group("tracker")
    TrackerArgs.add_args(tracker_group)

    dist_group = parser.add_argument_group("distributed")
    dist_group.add_argument("--dist-timeout", type=int, default=30)

    return parser.parse_args()


def resolve_cond_prefix_probs(args) -> Optional[list]:
    """CLI -> the objective's cond_prefix_probs (None: no prefix forcing)."""
    if getattr(args, "variable_prefix", False):
        return variable_prefix_probs(args.block_size)
    return None


def build_models(
    args, graft: bool = False
) -> Tuple[DFlashTargetModel, DFlashDraftModel]:
    """Build target model (backend wrapper) and draft model."""
    print_on_rank0(
        f"Loading target model from {args.target_model_path} using {args.target_model_backend} backend"
    )

    target_model_kwargs = {}

    target_model = get_dflash_target_model(
        pretrained_model_name_or_path=args.target_model_path,
        backend=args.target_model_backend,
        torch_dtype=torch.bfloat16,
        device="cuda" if args.target_model_backend == "hf" else None,
        trust_remote_code=args.trust_remote_code,
        **target_model_kwargs,
    )

    if args.draft_config_path:
        draft_config = AutoConfig.from_pretrained(
            args.draft_config_path, trust_remote_code=args.trust_remote_code
        )
        print_on_rank0(f"Loaded draft config from {args.draft_config_path}")
        # Warn if command-line args differ from config
        if (
            hasattr(draft_config, "block_size")
            and draft_config.block_size != args.block_size
        ):
            print_on_rank0(
                f"Warning: checkpoint block_size ({draft_config.block_size}) differs from "
                f"command-line arg ({args.block_size}). Using checkpoint value."
            )

    if not hasattr(draft_config, "dflash_config") or draft_config.dflash_config is None:
        draft_config.dflash_config = {}

    draft_config._attn_implementation = args.attention_backend
    print_on_rank0(f"Using attention backend: {args.attention_backend}")

    draft_model = DFlashDraftModel(draft_config).cuda().to(torch.bfloat16)

    target_model.set_capture_layers(draft_model.target_layer_ids)

    print_on_rank0(
        f"Draft config: block_size={draft_config.block_size}, "
        f"num_hidden_layers={draft_config.num_hidden_layers}, "
        f"num_target_layers={draft_config.num_target_layers}"
    )
    print_on_rank0(
        f"Draft model parameters: {sum(p.numel() for p in draft_model.parameters()):,}"
    )

    return target_model, draft_model


def build_dataloader(args, tokenizer) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Build train and eval dataloaders."""
    import hashlib

    cache_params_string = (
        f"{args.train_data_path}-"
        f"{args.max_length}-"
        f"{args.chat_template}-"
        f"{args.target_model_path}"
    )
    cache_key = hashlib.md5(cache_params_string.encode()).hexdigest()

    train_dataset = load_dataset("json", data_files=args.train_data_path)["train"]
    train_eagle3_dataset = build_eagle3_dataset(
        dataset=train_dataset,
        tokenizer=tokenizer,
        chat_template=args.chat_template,
        max_length=args.max_length,
        is_preformatted=args.is_preformatted,
        cache_dir=os.path.join(args.cache_dir, "processed_dataset"),
        cache_key=cache_key,
        num_proc=args.build_dataset_num_proc,
    )

    min_loss_tokens = 2 * args.block_size
    original_size = len(train_eagle3_dataset)
    train_eagle3_dataset = train_eagle3_dataset.filter(
        lambda x: x["loss_mask"].sum() >= min_loss_tokens
    )
    print_on_rank0(
        f"Filtered train dataset: {original_size} -> {len(train_eagle3_dataset)} samples"
    )

    train_dataloader = prepare_dp_dataloaders(
        train_eagle3_dataset,
        args.batch_size,
        num_workers=args.dataloader_num_workers,
        shuffle=True,
        process_group=get_dp_group(),
    )

    eval_dataloader = None

    return train_dataloader, eval_dataloader


def save_checkpoint(args, epoch, step, dflash_model, draft_model, optimizer):
    """Save checkpoint."""
    save_dir = os.path.join(args.output_dir, f"epoch_{epoch}_step_{step}")
    if dist.get_rank() == 0:
        os.makedirs(save_dir, exist_ok=True)
    dist.barrier()

    with FSDP.state_dict_type(dflash_model, StateDictType.FULL_STATE_DICT):
        state_dict = dflash_model.state_dict()
        draft_state_dict = {
            k.replace("draft_model.", ""): v
            for k, v in state_dict.items()
            if "draft_model." in k
        }

        if dist.get_rank() == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": step,
                    "args": args,
                    **optimizer.state_dict(),
                },
                os.path.join(save_dir, "training_state.pt"),
            )

            draft_model.save_pretrained(save_dir, state_dict=draft_state_dict)

            modeling_src = os.path.join(
                os.path.dirname(__file__),
                "modeling",
                "draft",
                "dflash.py",
            )
            modeling_dst = os.path.join(save_dir, "dflash.py")
            if os.path.exists(modeling_src):
                shutil.copy(modeling_src, modeling_dst)

            print_on_rank0(f"Saved checkpoint to {save_dir}")

            limit = getattr(args, "save_total_limit", 0)
            if limit and limit > 0:
                existing = [
                    d
                    for d in os.listdir(args.output_dir)
                    if d.startswith("epoch_") and "_step_" in d
                ]
                existing.sort(key=lambda d: int(d.split("_step_")[-1]))
                for old in existing[:-limit]:
                    shutil.rmtree(
                        os.path.join(args.output_dir, old), ignore_errors=True
                    )
                    print_on_rank0(f"Pruned old checkpoint {old}")

    dist.barrier()


def record_metrics(
    args,
    loss: float,
    accuracy: float,
    global_step: int,
    tracker,
    optimizer,
    train_dataloader=None,
    mode: str = "train",
    components=None,
    grad_norm=None,
) -> None:
    logdict = {}

    if mode == "train" and optimizer is not None:
        logdict["train/lr"] = optimizer.get_learning_rate()
    if grad_norm is not None:
        logdict["train/grad_norm"] = grad_norm

    logdict[f"{mode}/loss"] = loss
    logdict[f"{mode}/accuracy"] = accuracy
    if components:
        for ck, cv in components.items():
            logdict[f"{mode}/{ck}"] = cv

    print_on_rank0(
        f"{mode.capitalize()} - Step {global_step} [{global_step}/{args.num_epochs * len(train_dataloader) // args.accumulation_steps}?], Loss: {loss:.4f}, Acc: {accuracy:.4f}"
        + ("" if grad_norm is None else f", GradNorm: {grad_norm:.4f}")
    )

    tracker.log(logdict, step=global_step)


def main():

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logging.getLogger().setLevel(logging.INFO)
    warnings.filterwarnings(
        "ignore",
        "The .grad attribute of a Tensor that is not a leaf Tensor is being accessed",
    )

    args = parse_args()
    set_seed(args.seed)

    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)

    print_with_rank("Initialized distributed")

    draft_model_last_checkpoint = None
    ckpt_info = (0, 0)
    if args.resume and os.path.isdir(args.output_dir):
        draft_model_last_checkpoint, ckpt_info = get_last_checkpoint(args.output_dir)
        print(f"Last checkpoint detected: {draft_model_last_checkpoint}")

    # FORE-GRAFT: --resume beats --init-from -- a crashed graft run must continue its own
    # progress, not restart from the source drafter.
    if args.init_from is not None and not os.path.isdir(args.init_from):
        raise ValueError(f"--init-from {args.init_from!r} is not a directory")
    graft = args.init_from is not None and draft_model_last_checkpoint is None

    # If resuming, load config from checkpoint to ensure consistency
    if draft_model_last_checkpoint:
        checkpoint_config_path = os.path.join(
            draft_model_last_checkpoint, "config.json"
        )
        if os.path.exists(checkpoint_config_path):
            print(
                f"Loading draft config from checkpoint: {draft_model_last_checkpoint}"
            )
            args.draft_config_path = draft_model_last_checkpoint
    elif graft:
        args.draft_config_path = args.init_from

    target_model, draft_model = build_models(args, graft=graft)

    resume_state = None
    if draft_model_last_checkpoint or graft:
        weights_path = draft_model_last_checkpoint or args.init_from
        loaded_model = DFlashDraftModel.from_pretrained(
            weights_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=args.trust_remote_code,
        )
        draft_model.load_state_dict(loaded_model.state_dict())
        del loaded_model
        print(f"Loaded draft model weights from {weights_path}")

    if draft_model_last_checkpoint:
        training_state_path = os.path.join(
            draft_model_last_checkpoint, "training_state.pt"
        )
        if os.path.exists(training_state_path):
            resume_state = torch.load(
                training_state_path, map_location="cpu", weights_only=False
            )
            print(
                f"Will resume from epoch {resume_state['epoch']}, "
                f"step {resume_state['global_step']}"
            )
    elif graft:
        # FORE-GRAFT: PATH's own training_state.pt (if any) is deliberately not read.
        print_on_rank0(
            f"fine-tuning from {args.init_from}: weights loaded, optimizer fresh"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.target_model_path, trust_remote_code=args.trust_remote_code
    )

    if args.mask_token_id is not None:
        mask_token_id = args.mask_token_id
    elif (graft or draft_model_last_checkpoint) and draft_model.config.dflash_config.get(
        "mask_token_id"
    ) is not None:
        # FORE-GRAFT: the mask token is inherited from the source drafter (or checkpoint).
        mask_token_id = draft_model.config.dflash_config["mask_token_id"]
    elif tokenizer.mask_token_id is not None:
        mask_token_id = tokenizer.mask_token_id
    else:
        tokenizer.add_special_tokens({"mask_token": "<|MASK|>"})
        mask_token_id = tokenizer.mask_token_id
    print_on_rank0(f"Using mask_token_id: {mask_token_id}")

    draft_model.mask_token_id = mask_token_id
    draft_model.config.dflash_config["mask_token_id"] = mask_token_id
    draft_model.config.dflash_config["target_layer_ids"] = draft_model.target_layer_ids
    print_on_rank0(f"dflash_config: {draft_model.config.dflash_config}")

    train_dataloader, eval_dataloader = build_dataloader(args, tokenizer)

    steps_per_epoch = math.ceil(len(train_dataloader) / args.accumulation_steps)
    total_steps = args.num_epochs * steps_per_epoch
    print_on_rank0(f"Total training steps: {total_steps}")

    print_on_rank0("Loading target embeddings and head...")
    target_components = TargetEmbeddingsAndHead.from_pretrained(
        args.target_model_path,
        embed_key=args.embedding_key,
        lm_head_key=args.lm_head_key,
        device="cuda",
        trust_remote_code=args.trust_remote_code,
    )

    dflash_model = OnlineDFlashModel(
        draft_model=draft_model,
        target_lm_head=target_components.lm_head,
        target_embed_tokens=target_components.embed_tokens,
        block_size=draft_model.block_size,
        mask_token_id=mask_token_id,
        attention_backend=args.attention_backend,
        num_anchors=args.num_anchors,
        loss_type=args.loss_type,
        dpace_alpha=args.dpace_alpha,
        cond_prefix_probs=resolve_cond_prefix_probs(args),  # FORE-VARIABLE-PREFIX
    )

    dflash_model = FSDP(
        dflash_model,
        use_orig_params=True,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
    )
    print_with_rank("Initialized FSDP")

    start_epoch = ckpt_info[0]
    global_step = ckpt_info[1]

    optimizer = BF16Optimizer(
        draft_model,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        total_steps=total_steps,
    )

    if resume_state is not None:
        optimizer.load_state_dict(resume_state)
        start_epoch = resume_state["epoch"]
        global_step = resume_state["global_step"]
        del resume_state
        print_on_rank0(
            f"Restored optimizer/scheduler state: "
            f"epoch={start_epoch}, step={global_step}, "
            f"lr={optimizer.get_learning_rate():.6f}"
        )

    skip_steps = global_step - start_epoch * len(train_dataloader)

    print_on_rank0(f"Initializing tracker (report_to={args.report_to})...")
    tracker = create_tracker(args, args.output_dir)
    print_on_rank0("Tracker initialized successfully.")

    last_time = time.time()
    print_on_rank0(f"Starting training from epoch {start_epoch}, step {global_step}")

    for epoch in range(start_epoch, args.num_epochs):
        train_dataloader.sampler.set_epoch(epoch)
        draft_model.train()

        if dist.get_rank() == 0:
            progress_bar = tqdm(
                train_dataloader, desc=f"Training Epoch {epoch}", leave=True
            )
        else:
            progress_bar = train_dataloader

        for step_in_epoch, data in enumerate(progress_bar):
            if epoch == start_epoch and step_in_epoch < skip_steps:
                continue
            global_step += 1

            input_ids = data["input_ids"].cuda()
            attention_mask = data["attention_mask"].cuda()
            loss_mask = data["loss_mask"].cuda()
            target_output = target_model.generate_dflash_data(
                input_ids, attention_mask, loss_mask
            )
            hidden_states = target_output.hidden_states.cuda()  # Ensure on GPU

            loss, accuracy, components = dflash_model(
                input_ids=input_ids,
                hidden_states=hidden_states,
                loss_mask=loss_mask,
            )

            (loss / args.accumulation_steps).backward()

            if global_step % args.accumulation_steps == 0:
                optimizer.step()

            if global_step % args.log_interval == 0:
                loss_log = loss.clone()
                acc_log = accuracy.clone()
                dist.all_reduce(loss_log)
                dist.all_reduce(acc_log)
                loss_log = loss_log / dist.get_world_size()
                acc_log = acc_log / dist.get_world_size()

                comp_log = {}
                for ck, cv in components.items():
                    cvr = cv.clone()
                    dist.all_reduce(cvr)
                    comp_log[ck] = (cvr / dist.get_world_size()).item()

                grad_norm = None
                if optimizer.grad_norm is not None:
                    grad_sq = optimizer.grad_norm.detach().float().square()
                    dist.all_reduce(grad_sq)
                    grad_norm = grad_sq.sqrt().item()

                record_metrics(
                    args,
                    loss_log.item(),
                    acc_log.item(),
                    global_step,
                    tracker,
                    optimizer,
                    train_dataloader,
                    mode="train",
                    components=comp_log,
                    grad_norm=grad_norm,
                )

            if dist.get_rank() == 0:
                elapsed = time.time() - last_time
                last_time = time.time()
                progress_bar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "acc": f"{accuracy.item():.4f}",
                        "iter_time": f"{elapsed:.2f}s",
                    }
                )

            if global_step % args.save_interval == 0:
                save_checkpoint(
                    args, epoch, global_step, dflash_model, draft_model, optimizer
                )

    save_checkpoint(
        args, args.num_epochs, global_step, dflash_model, draft_model, optimizer
    )

    tracker.close()
    destroy_distributed()


if __name__ == "__main__":
    main()
