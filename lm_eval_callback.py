"""
LM-Eval Callback for Training

在训练过程中定期运行 lm-evaluation-harness 评估，监控模型能力变化。
用于检测灾难性遗忘（catastrophic forgetting）和模式塌缩。

Features:
- 轻量级评估：使用 --limit 限制样本数（固定取前 N 条，保证可比性）
- 复用 checkpoint：使用训练过程中已保存的 checkpoint，无需额外保存
- 双重记录：同时记录绝对分数和相对基线的 delta
- 自动上传：将评估结果上传到 WandB
"""

import os
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments


class LMEvalCallback(TrainerCallback):
    """
    在训练过程中定期运行 lm-eval 评估的 Callback
    
    Args:
        eval_steps: 每隔多少步运行一次评估（默认 100，必须是 save_steps 的倍数）
        tasks: 评估任务列表（默认 ['mmlu_stem', 'mmlu_other', 'gsm8k']）
        limit: 每个任务的样本数限制（默认 100，用于快速评估）
        num_fewshot: few-shot 样本数（默认 3）
        batch_size: 评估批次大小（默认 8）
        output_dir: 评估结果保存目录
        log_to_wandb: 是否记录到 WandB
    """
    
    def __init__(
        self,
        eval_steps: int = 100,  # Must be multiple of save_steps (100)
        tasks: List[str] = None,
        limit: int = 100,
        num_fewshot: int = 3,
        batch_size: int = 8,
        output_dir: str = None,
        log_to_wandb: bool = True,
    ):
        self.eval_steps = eval_steps
        self.tasks = tasks or ['mmlu_stem', 'mmlu_other', 'gsm8k']
        self.limit = limit
        self.num_fewshot = num_fewshot
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.log_to_wandb = log_to_wandb
        
        # 存储基线分数用于对比
        self.baseline_scores: Dict[str, float] = {}
        self.eval_history: List[Dict[str, Any]] = []
        
        # 记录上次评估步数，避免重复评估
        self.last_eval_step = -1
        
        # 状态文件名（用于断点恢复）
        self.state_filename = "lm_eval_state.json"
    
    def _get_state_path(self, output_dir: str) -> str:
        """获取状态文件路径"""
        return os.path.join(output_dir, self.state_filename)
    
    def _save_state(self, output_dir: str):
        """保存 callback 状态到文件（支持断点恢复）"""
        state = {
            "baseline_scores": self.baseline_scores,
            "eval_history": self.eval_history,
            "last_eval_step": self.last_eval_step,
            "tasks": self.tasks,
            "limit": self.limit,
        }
        
        state_path = self._get_state_path(output_dir)
        try:
            with open(state_path, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"⚠️ Failed to save LM-Eval state: {e}")
    
    def _load_state(self, output_dir: str) -> bool:
        """从文件加载 callback 状态（断点恢复时使用）"""
        state_path = self._get_state_path(output_dir)
        
        if not os.path.exists(state_path):
            return False
        
        try:
            with open(state_path, 'r') as f:
                state = json.load(f)
            
            self.baseline_scores = state.get("baseline_scores", {})
            self.eval_history = state.get("eval_history", [])
            self.last_eval_step = state.get("last_eval_step", -1)
            
            print(f"✅ Restored LM-Eval state from {state_path}")
            print(f"   Last eval step: {self.last_eval_step}")
            print(f"   Baseline scores: {self.baseline_scores}")
            print(f"   Eval history: {len(self.eval_history)} records")
            
            return True
        except Exception as e:
            print(f"⚠️ Failed to load LM-Eval state: {e}")
            return False
        
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs
    ):
        """
        在 checkpoint 保存之后运行 LM-Eval 评估
        
        使用 on_save 而不是 on_step_end，确保 checkpoint 已保存后再评估
        """
        
        # 只在主进程上运行评估
        if args.local_rank not in [-1, 0]:
            return
        
        current_step = state.global_step
        
        # 避免重复评估
        if current_step == self.last_eval_step:
            return
        
        # 检查是否到达 lm_eval 评估点
        if current_step % self.eval_steps != 0:
            return
        
        self.last_eval_step = current_step
        
        print(f"\n{'='*60}")
        print(f"🔍 Running LM-Eval at step {current_step} (checkpoint just saved)")
        print(f"{'='*60}")
        
        try:
            # 运行评估
            results = self._run_evaluation(model, args, state)
            
            if results:
                # 记录结果
                self._log_results(results, current_step, args)
                
                # 打印摘要
                self._print_summary(results, current_step)
                
        except Exception as e:
            print(f"⚠️ LM-Eval evaluation failed: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"{'='*60}\n")
    
    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs
    ):
        """训练开始时设置基线分数，或从断点恢复状态"""
        
        # 只在主进程上运行
        if args.local_rank not in [-1, 0]:
            return
        
        eval_output_dir = self.output_dir or args.output_dir
        
        # 尝试从文件恢复状态（断点恢复）
        if state.global_step > 0:
            print(f"\n{'='*60}")
            print(f"📊 Resuming from step {state.global_step}, restoring LM-Eval state...")
            print(f"{'='*60}")
            
            if self._load_state(eval_output_dir):
                # 成功恢复状态
                print(f"{'='*60}\n")
                return
            else:
                # 恢复失败，使用默认基线
                print(f"⚠️ No saved state found, using default baselines")
        
        print(f"\n{'='*60}")
        print(f"📊 Setting baseline scores (from pre-evaluated Qwen3-8B)")
        print(f"{'='*60}")
        
        # 使用已知的 Qwen3-8B 基线分数（完整评估的结果）
        # 这些是之前使用完整数据集评估得到的结果
        # 比在训练前运行 --limit 评估更准确
        known_baselines = {
            'mmlu_stem': 0.751,      # MMLU STEM: 75.1%
            'mmlu_other': 0.779,     # MMLU Other: 77.9%
            'gsm8k': 0.8802,         # GSM8K: 88.02%
        }
        
        for task in self.tasks:
            task_key = task.split(',')[0] if ',' in task else task
            if task_key in known_baselines:
                self.baseline_scores[task_key] = known_baselines[task_key]
                print(f"   {task_key}: {known_baselines[task_key]:.1%} (baseline)")
            else:
                print(f"   ⚠️ {task_key}: No baseline available")
        
        # 保存初始状态（支持断点恢复）
        self._save_state(eval_output_dir)
        
        print(f"\n📝 Note: Baselines from full evaluation of Qwen3-8B base model")
        print(f"   Training evaluations use --limit {self.limit} for speed")
        print(f"   Delta values show relative change from these baselines")
        print(f"{'='*60}\n")
    
    def _run_evaluation(
        self,
        model,
        args: TrainingArguments,
        state: TrainerState,
    ) -> Dict[str, float]:
        """
        运行 lm-eval 评估
        
        复用训练过程中已保存的 checkpoint（由 Trainer 的 save_steps 控制）
        lm_eval 的 --limit 参数固定取前 N 条数据，保证每次评估的样本一致
        """
        
        results = {}
        
        # 获取基础模型路径
        base_model_path = getattr(model.config, '_name_or_path', None)
        if not base_model_path:
            base_model_path = os.environ.get('BASE_MODEL_PATH', '/mnt/input/models/Qwen3-8B')
        
        eval_output_dir = self.output_dir or args.output_dir
        
        # 查找训练保存的最新 checkpoint
        # Trainer 按 save_steps 保存 checkpoint-{step} 目录
        checkpoint_dir = self._find_latest_checkpoint(eval_output_dir, state.global_step)
        
        if checkpoint_dir is None:
            print(f"⚠️ No checkpoint found at step {state.global_step}, skipping evaluation")
            print(f"   (Checkpoints are saved every {args.save_steps} steps)")
            return results
        
        try:
            # 检测 checkpoint 类型并构建正确的 model_args
            # FSDP 和标准 LoRA 保存格式不同
            model_args = self._build_model_args(base_model_path, checkpoint_dir)
            
            eval_result_dir = os.path.join(eval_output_dir, f"_eval_results_step_{state.global_step}")
            os.makedirs(eval_result_dir, exist_ok=True)
            
            # 使用单 GPU 评估，避免与 FSDP 训练进程冲突
            # FSDP 训练已经占用了所有 GPU 的分布式环境
            # 在回调中启动另一个 multi-GPU 进程会导致冲突
            # 单 GPU + device_map=auto 已经足够快（100 samples）
            cmd = [
                "lm_eval",
                "--model", "hf",
                "--model_args", model_args,
                "--tasks", ",".join(self.tasks),
                "--num_fewshot", str(self.num_fewshot),
                "--batch_size", str(self.batch_size),
                "--limit", str(self.limit),
                "--output_path", eval_result_dir,
            ]
            print(f"   🔄 Single-process evaluation (avoiding FSDP conflict)")
            
            print(f"   Checkpoint: {os.path.basename(checkpoint_dir)}")
            print(f"   Tasks: {', '.join(self.tasks)}")
            print(f"   Limit: {self.limit} samples per task (fixed, first N samples)")
            print(f"   Few-shot: {self.num_fewshot}")
            print(f"   Command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 分钟超时
            )
            
            if result.returncode != 0:
                print(f"⚠️ lm_eval failed with return code {result.returncode}")
                print(f"   STDOUT (last 2000 chars):")
                print(result.stdout[-2000:] if result.stdout else "(empty)")
                print(f"   STDERR (last 2000 chars):")
                print(result.stderr[-2000:] if result.stderr else "(empty)")
                return results
            
            results = self._parse_results(eval_result_dir)
            
        except subprocess.TimeoutExpired:
            print("⚠️ LM-Eval timed out after 30 minutes")
        except Exception as e:
            print(f"⚠️ LM-Eval error: {e}")
        
        return results
    
    def _build_model_args(self, base_model_path: str, checkpoint_dir: str) -> str:
        """
        根据 checkpoint 类型构建 lm_eval 的 model_args
        
        支持的 checkpoint 类型：
        1. 标准 LoRA adapter (adapter_config.json): 使用 peft 参数加载
        2. FSDP 保存的 LoRA (pytorch_model_fsdp_0): 需要合并后加载
        3. 完整模型 checkpoint (model.safetensors): 直接加载
        """
        
        # 检测 checkpoint 类型
        has_adapter_config = os.path.exists(os.path.join(checkpoint_dir, "adapter_config.json"))
        has_fsdp_sharded = os.path.exists(os.path.join(checkpoint_dir, "pytorch_model_fsdp_0"))
        has_adapter_safetensors = os.path.exists(os.path.join(checkpoint_dir, "adapter_model.safetensors"))
        has_adapter_bin = os.path.exists(os.path.join(checkpoint_dir, "adapter_model.bin"))
        
        print(f"   📁 Checkpoint type detection:")
        print(f"      adapter_config.json: {has_adapter_config}")
        print(f"      adapter_model.safetensors: {has_adapter_safetensors}")
        print(f"      adapter_model.bin: {has_adapter_bin}")
        print(f"      pytorch_model_fsdp_0 (FSDP): {has_fsdp_sharded}")
        
        # 1. 标准 PEFT LoRA adapter（最常见的情况）
        if has_adapter_config and (has_adapter_safetensors or has_adapter_bin):
            print(f"   ✅ Using standard PEFT adapter loading")
            return f"pretrained={base_model_path},peft={checkpoint_dir},trust_remote_code=True,device_map=auto"
        
        # 2. FSDP sharded checkpoint - 这种情况比较复杂
        # FSDP with LoRA 会保存完整的模型状态（包括 LoRA 权重）
        # lm_eval 不直接支持这种格式，需要先手动加载并保存为标准格式
        if has_fsdp_sharded:
            print(f"   ⚠️ FSDP sharded checkpoint detected")
            print(f"   📝 FSDP checkpoints require manual loading and conversion")
            print(f"   Attempting to use base model + FSDP merged weights...")
            
            # 对于 FSDP，lm_eval 无法直接加载分片 checkpoint
            # 方案：检查是否有 FSDP 保存的完整模型
            # 如果有 model.safetensors 或 pytorch_model.bin，可以直接用
            if os.path.exists(os.path.join(checkpoint_dir, "model.safetensors")):
                print(f"   ✅ Found merged model.safetensors, loading directly")
                return f"pretrained={checkpoint_dir},trust_remote_code=True,device_map=auto"
            elif os.path.exists(os.path.join(checkpoint_dir, "pytorch_model.bin")):
                print(f"   ✅ Found merged pytorch_model.bin, loading directly")
                return f"pretrained={checkpoint_dir},trust_remote_code=True,device_map=auto"
            else:
                # FSDP sharded 格式，lm_eval 无法直接处理
                print(f"   ⚠️ FSDP sharded format not directly supported by lm_eval")
                print(f"   Falling back to base model evaluation (no LoRA)")
                return f"pretrained={base_model_path},trust_remote_code=True,device_map=auto"
        
        # 3. 完整模型 checkpoint（已合并的模型）
        if os.path.exists(os.path.join(checkpoint_dir, "model.safetensors")):
            print(f"   ✅ Using full model checkpoint")
            return f"pretrained={checkpoint_dir},trust_remote_code=True,device_map=auto"
        
        # 4. Fallback: 只有 adapter_config.json 但没有权重文件（不应该发生）
        if has_adapter_config:
            print(f"   ⚠️ adapter_config.json found but no weights, trying anyway...")
            return f"pretrained={base_model_path},peft={checkpoint_dir},trust_remote_code=True,device_map=auto"
        
        # 5. 无法识别的格式
        print(f"   ⚠️ Unknown checkpoint format, using base model only")
        return f"pretrained={base_model_path},trust_remote_code=True,device_map=auto"

    def _find_latest_checkpoint(
        self,
        output_dir: str,
        current_step: int,
    ) -> Optional[str]:
        """
        查找最近的已保存 checkpoint
        
        Trainer 保存的 checkpoint 格式：checkpoint-{step}
        返回与当前步数最接近的 checkpoint 路径
        """
        checkpoints = []
        
        if not os.path.exists(output_dir):
            return None
        
        for name in os.listdir(output_dir):
            if name.startswith("checkpoint-"):
                try:
                    step = int(name.split("-")[1])
                    checkpoint_path = os.path.join(output_dir, name)
                    # 支持多种 checkpoint 格式：
                    # 1. 标准 LoRA: adapter_config.json
                    # 2. FSDP sharded: pytorch_model_fsdp_0 目录
                    # 3. FSDP full state: pytorch_model.bin 或 model.safetensors
                    # 4. 任何 trainer_state.json（Trainer 总是保存这个）
                    checkpoint_markers = [
                        "adapter_config.json",          # Standard LoRA
                        "adapter_model.safetensors",    # LoRA safetensors
                        "pytorch_model_fsdp_0",         # FSDP sharded
                        "model.safetensors",            # Full model
                        "pytorch_model.bin",            # PyTorch format
                        "trainer_state.json",           # Trainer always saves this
                    ]
                    
                    is_valid = any(
                        os.path.exists(os.path.join(checkpoint_path, marker))
                        for marker in checkpoint_markers
                    )
                    
                    if is_valid:
                        checkpoints.append((step, checkpoint_path))
                except (ValueError, IndexError):
                    continue
        
        if not checkpoints:
            return None
        
        # 找到不超过当前步数的最近 checkpoint
        valid_checkpoints = [(s, p) for s, p in checkpoints if s <= current_step]
        if valid_checkpoints:
            return max(valid_checkpoints, key=lambda x: x[0])[1]
        
        # 如果没有，返回最早的（用于 baseline）
        return min(checkpoints, key=lambda x: x[0])[1]
    
    def _parse_results(self, result_dir: str) -> Dict[str, float]:
        """解析 lm_eval 输出的结果文件"""
        results = {}
        
        # 查找 results.json 文件
        for pattern in ["results.json", "**/results*.json"]:
            matches = list(Path(result_dir).glob(pattern))
            if matches:
                try:
                    with open(matches[0]) as f:
                        data = json.load(f)
                    
                    if "results" in data:
                        for task, task_data in data["results"].items():
                            # 提取主要指标
                            score = None
                            for metric in [
                                "acc", "acc_norm", "exact_match",
                                "acc,none", "acc_norm,none",
                                "exact_match,flexible-extract",
                                "exact_match,strict-match",
                            ]:
                                if metric in task_data:
                                    score = task_data[metric]
                                    break
                            
                            if score is not None:
                                # 简化任务名
                                task_name = task.split(",")[0] if "," in task else task
                                results[task_name] = score
                    
                    break
                except Exception as e:
                    print(f"⚠️ Failed to parse results: {e}")
        
        return results
    
    def _log_results(
        self,
        results: Dict[str, float],
        step: int,
        args: TrainingArguments,
        is_baseline: bool = False,
    ):
        """记录评估结果到 WandB 和本地"""
        
        # 保存到历史
        record = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "is_baseline": is_baseline,
            "results": results,
        }
        
        # 添加与基线的对比
        if not is_baseline and self.baseline_scores:
            deltas = {}
            for task, score in results.items():
                if task in self.baseline_scores:
                    delta = score - self.baseline_scores[task]
                    deltas[f"{task}_delta"] = delta
            record["deltas"] = deltas
        
        self.eval_history.append(record)
        
        # 保存到本地文件
        eval_output_dir = self.output_dir or args.output_dir
        history_file = os.path.join(eval_output_dir, "lm_eval_history.json")
        try:
            with open(history_file, 'w') as f:
                json.dump(self.eval_history, f, indent=2)
        except Exception as e:
            print(f"⚠️ Failed to save eval history: {e}")
        
        # 保存 callback 状态（支持断点恢复）
        self._save_state(eval_output_dir)
        
        # 记录到 WandB
        if self.log_to_wandb:
            try:
                import wandb
                if wandb.run is not None:
                    log_dict = {}
                    
                    # 始终记录绝对值（主要指标）
                    for task, score in results.items():
                        log_dict[f"lm_eval/{task}"] = score
                    
                    # 如果有基线，也记录 delta（辅助指标）
                    if not is_baseline and self.baseline_scores:
                        for task, score in results.items():
                            if task in self.baseline_scores:
                                delta = score - self.baseline_scores[task]
                                log_dict[f"lm_eval/{task}_delta"] = delta
                        
                        # 添加综合 delta 指标（所有任务的平均变化）
                        deltas = [
                            score - self.baseline_scores[task]
                            for task, score in results.items()
                            if task in self.baseline_scores
                        ]
                        if deltas:
                            log_dict["lm_eval/avg_delta"] = sum(deltas) / len(deltas)
                    
                    wandb.log(log_dict, step=step)
            except Exception as e:
                print(f"⚠️ Failed to log to WandB: {e}")
    
    def _print_summary(
        self,
        results: Dict[str, float],
        step: int,
        is_baseline: bool = False,
    ):
        """打印评估结果摘要"""
        
        print(f"\n📊 LM-Eval Results (Step {step}):")
        print("-" * 50)
        
        if is_baseline:
            print(f"{'Task':<25} {'Score':>10}")
            print("-" * 50)
            for task, score in results.items():
                print(f"{task:<25} {score:>9.1%}")
        else:
            print(f"{'Task':<25} {'Score':>10} {'Change':>10}")
            print("-" * 50)
            
            has_forgetting = False
            for task, score in results.items():
                if task in self.baseline_scores:
                    delta = score - self.baseline_scores[task]
                    delta_str = f"{delta:+.1%}"
                    
                    # 检测遗忘
                    if delta < -0.05:  # 超过 5% 下降
                        status = "⚠️"
                        has_forgetting = True
                    elif delta < -0.02:  # 2-5% 下降
                        status = "📉"
                    elif delta > 0.02:
                        status = "📈"
                    else:
                        status = "✅"
                    
                    print(f"{task:<25} {score:>9.1%} {delta_str:>8} {status}")
                else:
                    print(f"{task:<25} {score:>9.1%} {'N/A':>10}")
            
            if has_forgetting:
                print("\n⚠️ WARNING: Significant capability degradation detected!")
                print("   Consider: reducing learning rate, using more regularization,")
                print("   or mixing in general capability data.")
        
        print("-" * 50)


def create_lm_eval_callback(
    eval_steps: int = 100,  # Must be multiple of save_steps
    tasks: List[str] = None,
    limit: int = 100,
    enabled: bool = True,
) -> Optional[LMEvalCallback]:
    """
    工厂函数：创建 LM-Eval Callback
    
    Args:
        eval_steps: 评估间隔步数
        tasks: 评估任务
        limit: 每个任务的样本限制
        enabled: 是否启用
    
    Returns:
        LMEvalCallback 实例或 None
    """
    if not enabled:
        return None
    
    # 检查 lm_eval 是否可用
    try:
        result = subprocess.run(
            ["lm_eval", "--help"],
            capture_output=True,
            timeout=10,
        )
        if result.returncode != 0:
            print("⚠️ lm_eval not available, disabling LM-Eval callback")
            return None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("⚠️ lm_eval not found, disabling LM-Eval callback")
        return None
    
    return LMEvalCallback(
        eval_steps=eval_steps,
        tasks=tasks or ['mmlu_stem', 'mmlu_other', 'gsm8k'],
        limit=limit,
    )
