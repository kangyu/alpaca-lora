import os
import sys
from typing import List
import glob
from pathlib import Path

import fire
import torch
import transformers
from datasets import load_dataset

# Suppress c10d IPv6 warnings in distributed training
os.environ.setdefault('GLOO_SOCKET_IFNAME', 'eth0')
os.environ.setdefault('NCCL_SOCKET_IFNAME', 'eth0')

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.prompter import Prompter
from azureml_callback import AzureMLCallback
from lm_eval_callback import LMEvalCallback


def find_latest_checkpoint(output_dir: str) -> str:
    """Auto-detect the latest valid checkpoint in the output directory.
    
    Each experiment has its own unique output_dir (e.g., /mnt/output/lora_qwen3_xxx_timestamp/),
    so all checkpoints in that directory belong to the current experiment.
    
    Args:
        output_dir: Output directory for current experiment (contains checkpoint-* subdirs)
        
    Returns:
        Path to the latest valid checkpoint, or None if not found
    """
    if not os.path.exists(output_dir):
        return None
    
    # Find all checkpoint-* directories in the current experiment's output directory
    checkpoint_dirs = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not checkpoint_dirs:
        return None
    
    # Filter for valid training checkpoints (must have trainer_state.json)
    valid_checkpoints = []
    for checkpoint_dir in checkpoint_dirs:
        trainer_state_path = os.path.join(checkpoint_dir, "trainer_state.json")
        
        # Must have trainer_state.json to be a valid training checkpoint
        if not os.path.exists(trainer_state_path):
            continue
        
        # Verify required model files exist
        required_files = ["training_args.bin"]
        adapter_file_exists = False
        for f in ["adapter_model.safetensors", "adapter_model.bin", "pytorch_model.bin"]:
            if os.path.exists(os.path.join(checkpoint_dir, f)):
                adapter_file_exists = True
                break
        
        # Only include if all required files are present
        if adapter_file_exists and all(os.path.exists(os.path.join(checkpoint_dir, f)) for f in required_files):
            valid_checkpoints.append(checkpoint_dir)
    
    if not valid_checkpoints:
        return None
    
    # Sort by step number (checkpoint-200, checkpoint-400, etc.)
    def get_step_number(path):
        try:
            return int(os.path.basename(path).split("-")[-1])
        except (ValueError, IndexError):
            return -1
    
    valid_checkpoints.sort(key=get_step_number)
    latest_checkpoint = valid_checkpoints[-1]
    
    return latest_checkpoint


def train(
    # model/data params
    base_model: str = "",  # the only required argument
    data_path: str = "yahma/alpaca-cleaned",
    output_dir: str = "./lora-alpaca",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 2000,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
    auto_resume: bool = True,  # automatically resume from the latest checkpoint if available
    save_steps: int = 100,  # save checkpoint every N steps
    # LM-Eval monitoring
    lm_eval_enabled: bool = True,  # enable lm-eval during training
    lm_eval_steps: int = 100,  # run lm-eval every N steps (should match save_steps)
    lm_eval_tasks: List[str] = ["mmlu_stem", "gsm8k"],  # tasks to evaluate
    lm_eval_limit: int = 100,  # samples per task (for speed)
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"auto_resume: {auto_resume}\n"
            f"prompt template: {prompt_template_name}\n"
            f"lm_eval_enabled: {lm_eval_enabled}\n"
            f"lm_eval_steps: {lm_eval_steps}\n"
            f"lm_eval_tasks: {lm_eval_tasks}\n"
            f"lm_eval_limit: {lm_eval_limit}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    
    # 保存基础模型路径到环境变量，供 LM-Eval callback 使用
    os.environ['BASE_MODEL_PATH'] = base_model
    
    # Auto-detect latest checkpoint if auto_resume is enabled
    if auto_resume and not resume_from_checkpoint:
        detected_checkpoint = find_latest_checkpoint(output_dir)
        if detected_checkpoint:
            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                print(f"\n{'='*60}")
                print(f"🔄 AUTO-RESUME: Detected checkpoint at {detected_checkpoint}")
                print(f"{'='*60}\n")
            resume_from_checkpoint = detected_checkpoint
    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(prompt_template_name)

    # FSDP handles device placement automatically via accelerate
    # No need for device_map when using FSDP
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    # Load model with FSDP-compatible settings (no quantization, full precision LoRA)
    # Strategy:
    # 1. Try Flash Attention 2 (fastest)
    # 2. Fallback to SDPA (PyTorch built-in, most compatible)
    
    model = None
    tokenizer = None
    method_used = None
    
    # Determine compute dtype
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"\n{'='*60}")
        print(f"🚀 Loading model for FSDP training (full precision LoRA)")
        print(f"   Compute dtype: {compute_dtype}")
        print(f"   World size: {world_size}")
        print(f"{'='*60}")
    
    # Strategy 1: Try Flash Attention 2
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"\n🔄 Attempting to load with Flash Attention 2...")
    
    try:
        import flash_attn
        
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=compute_dtype,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            use_cache=False,  # Disable KV cache for training
        )
        
        method_used = "Flash Attention 2 + LoRA"
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"✅ Successfully loaded with Flash Attention 2!")
    
    except Exception as e:
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"⚠️  Flash Attention 2 failed: {str(e)}")
    
    # Strategy 2: Fallback to SDPA
    if model is None:
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"🔄 Falling back to SDPA (PyTorch built-in)...")
        
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=compute_dtype,
            trust_remote_code=True,
            attn_implementation="sdpa",
            use_cache=False,  # Disable KV cache for training
        )
        
        method_used = "SDPA + LoRA"
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"✅ Successfully loaded with SDPA!")
    
    # Enable input gradients for LoRA training
    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"\n{'='*60}")
        print(f"🎯 Training method: {method_used}")
        print(f"   FSDP mode: {ddp}")
        print(f"   World size: {world_size}")
        print(f"   Gradient checkpointing: enabled")
        print(f"{'='*60}\n")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    # Apply LoRA configuration
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        # Use RSLoRA for better performance on larger rank values
        use_rslora=lora_r >= 16,
    )
    model = get_peft_model(model, lora_config)
    
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"\n✅ Applied LoRA configuration:")
        print(f"   Rank: {lora_r}, Alpha: {lora_alpha}")
        print(f"   Target modules: {lora_target_modules}")
        print(f"   Trainable params: {model.print_trainable_parameters() or 'see above'}")

    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    # Handle checkpoint resumption
    # Hugging Face Trainer can automatically load optimizer, scheduler, and training state
    # We only need to manually load adapter weights if it's a final adapter (not a training checkpoint)
    if resume_from_checkpoint:
        # Check if it's a training checkpoint (has trainer_state.json) or final adapter
        is_training_checkpoint = os.path.exists(
            os.path.join(resume_from_checkpoint, "trainer_state.json")
        )
        
        if is_training_checkpoint:
            # Full training checkpoint - let Trainer handle everything
            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                print(f"\n✅ Loading training checkpoint from: {resume_from_checkpoint}")
                print(f"   This will restore: model weights, optimizer, scheduler, and training state\n")
            # Keep resume_from_checkpoint as-is for trainer.train()
        else:
            # Final adapter only - manually load weights
            # Try different file formats: .safetensors (new) or .bin (legacy)
            adapter_files = [
                "adapter_model.safetensors",
                "adapter_model.bin",
                "pytorch_model.bin"
            ]
            
            checkpoint_loaded = False
            for adapter_file in adapter_files:
                checkpoint_path = os.path.join(resume_from_checkpoint, adapter_file)
                if os.path.exists(checkpoint_path):
                    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                        print(f"\n✅ Loading adapter weights from: {checkpoint_path}")
                    
                    if adapter_file.endswith(".safetensors"):
                        from safetensors.torch import load_file
                        adapters_weights = load_file(checkpoint_path)
                    else:
                        adapters_weights = torch.load(checkpoint_path)
                    
                    set_peft_model_state_dict(model, adapters_weights)
                    checkpoint_loaded = True
                    
                    # Don't pass to trainer.train() since we already loaded weights
                    resume_from_checkpoint = False
                    break
            
            if not checkpoint_loaded:
                print(f"\n⚠️  WARNING: No valid adapter checkpoint found in {resume_from_checkpoint}")
                print(f"    Looked for: {', '.join(adapter_files)}")
                resume_from_checkpoint = False

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    # Note: When using FSDP via accelerate, we don't need manual parallelization
    # accelerate handles device placement and sharding automatically

    # Determine mixed precision settings
    use_bf16 = torch.cuda.is_bf16_supported()
    
    # 构建 callbacks 列表
    callbacks = [AzureMLCallback()]
    
    # 添加 LM-Eval callback（用于监控模型能力变化）
    if lm_eval_enabled:
        lm_eval_callback = LMEvalCallback(
            eval_steps=lm_eval_steps,
            tasks=lm_eval_tasks if isinstance(lm_eval_tasks, list) else lm_eval_tasks.split(','),
            limit=lm_eval_limit,
            output_dir=output_dir,
            log_to_wandb=use_wandb,
        )
        callbacks.append(lm_eval_callback)
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"\n📊 LM-Eval monitoring enabled:")
            print(f"   Tasks: {lm_eval_tasks}")
            print(f"   Eval every {lm_eval_steps} steps")
            print(f"   {lm_eval_limit} samples per task\n")
    
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=use_bf16,
            fp16=not use_bf16,
            logging_steps=10,
            optim="adamw_torch",
            eval_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=save_steps if val_set_size > 0 else None,
            save_steps=save_steps,
            output_dir=output_dir,
            save_total_limit=5,  # Keep more checkpoints for evaluation history
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
            # FSDP settings are handled by accelerate config, but we can add hints
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks=callbacks,
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    # Apply torch.compile for additional speedup (PyTorch 2.0+)
    # Use reduce-overhead mode for best performance with gradient checkpointing
    if torch.__version__ >= "2" and sys.platform != "win32":
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print("\n⚡ Applying torch.compile with reduce-overhead mode...")
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                print(f"⚠️ torch.compile failed, continuing without compilation: {e}")

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"\n🚀 Starting training from {'checkpoint' if resume_from_checkpoint else 'scratch'}...")
    
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


if __name__ == "__main__":
    fire.Fire(train)
