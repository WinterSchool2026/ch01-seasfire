"""
Simple callbacks for the SeasFire challenge (no WandB dependency).
"""

import lightning.pytorch as pl


class PrintMetricsCallback(pl.Callback):
    """Print key metrics at the end of each validation epoch."""

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        parts = [f"Epoch {epoch:3d}"]
        for key in ["val/loss", "val/auprc"]:
            if key in metrics:
                parts.append(f"{key}: {metrics[key]:.4f}")
        print(" | ".join(parts))
