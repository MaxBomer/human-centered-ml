"""Utilities for Weights & Biases logging in deep active learning experiments."""
from __future__ import annotations

import wandb
from typing import Any, Callable, Optional


def initialize_wandb_run(
    use_wandb: bool,
    project: str,
    entity: Optional[str],
    run_name: str,
    config: dict[str, Any]
) -> Optional[wandb.run]:
    """
    Initialize a W&B run with the given configuration.
    
    Args:
        use_wandb: Whether to enable W&B logging.
        project: W&B project name.
        entity: W&B entity (username or team); None uses default.
        run_name: Human-readable name for this run.
        config: Dict of hyperparameters and experiment config to log.
    
    Returns:
        The W&B run object if enabled, else None.
    """
    if not use_wandb:
        return None
    
    try:
        run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            config=config,
            reinit=False
        )
        return run
    except Exception as e:
        print(f"Warning: Failed to initialize W&B: {e}")
        return None


def log_round_metrics(
    run: Optional[wandb.run],
    round_number: int,
    test_accuracy: float,
    num_labeled: int,
    acquisition_time: float,
    **kwargs: Any
) -> None:
    """
    Log per-round AL metrics to W&B.
    
    Args:
        run: W&B run object (or None if logging disabled).
        round_number: Active learning round number.
        test_accuracy: Test set accuracy for this round.
        num_labeled: Number of labeled samples at this round.
        acquisition_time: Time taken to acquire samples (seconds).
        **kwargs: Additional metrics to log.
    """
    if run is None:
        return
    
    metrics = {
        'round': round_number,
        'test_accuracy': test_accuracy,
        'num_labeled': num_labeled,
        'acquisition_time_sec': acquisition_time,
    }
    metrics.update(kwargs)
    
    try:
        wandb.log(metrics, step=round_number)
    except Exception as e:
        print(f"Warning: Failed to log metrics to W&B: {e}")


def log_train_epoch_metrics(
    run: Optional[wandb.run],
    round_number: int,
    epoch: int,
    train_loss: float,
    val_loss: Optional[float] = None,
    **kwargs: Any
) -> None:
    """
    Log per-epoch training metrics to W&B.
    
    Args:
        run: W&B run object (or None if logging disabled).
        round_number: Active learning round number.
        epoch: Training epoch within this round.
        train_loss: Training loss for this epoch.
        val_loss: Validation loss for this epoch (optional).
        **kwargs: Additional metrics to log.
    """
    if run is None:
        return
    
    metrics = {
        'round': round_number,
        'epoch': epoch,
        'train_loss': train_loss,
    }
    if val_loss is not None:
        metrics['val_loss'] = val_loss
    metrics.update(kwargs)
    
    try:
        global_step = round_number * 1000 + epoch
        wandb.log(metrics, step=global_step)
    except Exception as e:
        print(f"Warning: Failed to log epoch metrics to W&B: {e}")


def log_run_summary(
    run: Optional[wandb.run],
    summary_dict: dict[str, Any]
) -> None:
    """
    Log run-level summary statistics (AUBC, final accuracy, etc.) to W&B.
    
    Args:
        run: W&B run object (or None if logging disabled).
        summary_dict: Dict of summary statistics.
    """
    if run is None:
        return
    
    try:
        for key, value in summary_dict.items():
            run.summary[key] = value
    except Exception as e:
        print(f"Warning: Failed to log run summary to W&B: {e}")


def get_wandb_log_callback(
    run: Optional[wandb.run],
    round_number: int
) -> Optional[Callable[[int, float, Optional[float]], None]]:
    """
    Create a callback function for logging per-epoch metrics from net.train().
    
    Args:
        run: W&B run object (or None).
        round_number: Current AL round number.
    
    Returns:
        A callable that logs (epoch, train_loss, val_loss) or None if W&B disabled.
    """
    if run is None:
        return None
    
    def callback(epoch: int, train_loss: float, val_loss: Optional[float] = None) -> None:
        log_train_epoch_metrics(run, round_number, epoch, train_loss, val_loss)
    
    return callback


def finalize_wandb_run(run: Optional[wandb.run]) -> None:
    """
    Finalize (finish) a W&B run.
    
    Args:
        run: W&B run object (or None if logging disabled).
    """
    if run is None:
        return
    
    try:
        wandb.finish()
    except Exception as e:
        print(f"Warning: Failed to finalize W&B run: {e}")

