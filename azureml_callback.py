"""Azure ML Callback for Hugging Face Trainer to log metrics."""
import os
from transformers import TrainerCallback

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class AzureMLCallback(TrainerCallback):
    """Callback to log metrics to Azure ML using MLflow."""
    
    def __init__(self):
        self.mlflow_available = MLFLOW_AVAILABLE
        self.run_started = False
        
        if self.mlflow_available:
            # MLflow is automatically configured by Azure ML
            tracking_uri = mlflow.get_tracking_uri()
            print(f"✅ MLflow is available. Tracking URI: {tracking_uri}")
        else:
            print("⚠️ MLflow is not available. Azure ML metrics will not be logged. Install with: pip install azureml-mlflow")
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Log training parameters using existing Azure ML run."""
        if not self.mlflow_available:
            return
        
        # Only log from rank 0 to avoid duplicates in distributed training
        import os as _os
        local_rank = int(_os.environ.get('LOCAL_RANK', 0))
        if local_rank != 0:
            return
        
        try:
            # In Azure ML, there's already an active run created by the platform
            active_run = mlflow.active_run()
            if active_run is not None:
                print(f"Using existing Azure ML run: {active_run.info.run_id}")
            else:
                # This shouldn't happen in Azure ML, but start a run as fallback
                mlflow.start_run()
                self.run_started = True
                print("Started new MLflow run")
            
            # Set hierarchical tags from environment variables
            project = os.environ.get('AML_PROJECT', 'alpaca-lora')
            experiment = os.environ.get('AML_EXPERIMENT', 'default')
            trial = os.environ.get('AML_TRIAL', 'trial')
            
            mlflow.set_tags({
                'project': project,
                'experiment': experiment,
                'trial': trial,
                'framework': 'alpaca-lora',
                'model': 'qwen3-8b',
            })
            print(f"Set MLflow tags - project: {project}, experiment: {experiment}, trial: {trial}")
            
            # Log hyperparameters
            params = {
                'learning_rate': args.learning_rate,
                'batch_size': args.per_device_train_batch_size * args.gradient_accumulation_steps,
                'micro_batch_size': args.per_device_train_batch_size,
                'num_epochs': args.num_train_epochs,
                'warmup_steps': args.warmup_steps,
                'max_steps': args.max_steps if args.max_steps > 0 else -1,
            }
            mlflow.log_params(params)
            print(f"Logged training parameters to Azure ML")
        except Exception as e:
            print(f"Warning: Failed to log params to Azure ML: {e}")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log metrics to Azure ML after each logging step."""
        if not self.mlflow_available or logs is None:
            return
        
        # Only log from rank 0 to avoid duplicates
        import os as _os
        local_rank = int(_os.environ.get('LOCAL_RANK', 0))
        if local_rank != 0:
            return
        
        # Debug: print all available metrics
        print(f"[MLflow Debug] Available logs keys: {list(logs.keys())}")
        
        # Separate metrics by type and add prefixes like WandB does
        train_metrics = {}
        eval_metrics = {}
        other_metrics = {}
        
        for key, value in logs.items():
            if not isinstance(value, (int, float)):
                continue
            
            # Check if it's an eval metric
            if key.startswith('eval_'):
                # Add eval/ prefix to match WandB format
                new_key = f"eval/{key[5:]}"  # Remove 'eval_' and add 'eval/'
                eval_metrics[new_key] = value
            elif key in ['loss', 'learning_rate', 'grad_norm', 'epoch', 'train_loss', 'train_runtime', 
                         'train_samples_per_second', 'train_steps_per_second']:
                # These are training metrics
                if key.startswith('train_'):
                    # Already has train_ prefix, convert to train/
                    new_key = f"train/{key[6:]}"
                else:
                    # Add train/ prefix
                    new_key = f"train/{key}"
                train_metrics[new_key] = value
            else:
                # Keep other metrics as-is
                other_metrics[key] = value
        
        # Combine all metrics
        all_metrics = {**train_metrics, **eval_metrics, **other_metrics}
        
        if all_metrics:
            print(f"[MLflow Debug] Logging {len(all_metrics)} metrics: {list(all_metrics.keys())}")
            # Log to MLflow (Azure ML backend)
            try:
                mlflow.log_metrics(all_metrics, step=state.global_step)
            except Exception as e:
                print(f"Warning: Failed to log metrics to Azure ML: {e}")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Log final metrics. Don't end the run as it's managed by Azure ML."""
        if not self.mlflow_available:
            return
        
        # Only log from rank 0
        import os as _os
        local_rank = int(_os.environ.get('LOCAL_RANK', 0))
        if local_rank != 0:
            return
        
        try:
            if state.best_metric is not None:
                mlflow.log_metric("best_metric", state.best_metric)
            print("Logged final metrics to Azure ML")
            
            # Don't call mlflow.end_run() - let Azure ML manage the run lifecycle
        except Exception as e:
            print(f"Warning: Failed to finalize Azure ML logging: {e}")
