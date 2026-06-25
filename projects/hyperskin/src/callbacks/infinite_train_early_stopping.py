import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from typing import Optional


class InfiniteTrainEarlyStopping(EarlyStopping):
    def __init__(
        self,
        monitor: str,
        check_every_n_steps: int,
        min_delta: float = 0.0,
        patience: int = 3,
        verbose: bool = False,
        mode: str = "min",
        strict: bool = True,
        check_finite: bool = True,
        stopping_threshold: Optional[float] = None,
        divergence_threshold: Optional[float] = None,
        log_rank_zero_only: bool = False,
    ):
        super().__init__(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            verbose=verbose,
            mode=mode,
            strict=strict,
            check_finite=check_finite,
            stopping_threshold=stopping_threshold,
            divergence_threshold=divergence_threshold,
            check_on_train_epoch_end=False,
            log_rank_zero_only=log_rank_zero_only,
        )
        self.check_every_n_steps = check_every_n_steps

    # Disable epoch/validation hooks
    def on_train_epoch_end(self, trainer, pl_module):
        pass

    def on_validation_end(self, trainer, pl_module):
        pass

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # We check if check_every_n_steps is defined and greater than 0
        if self.check_every_n_steps > 0:
            # Check if the current global step aligns with the interval
            if trainer.global_step % self.check_every_n_steps == 0:
                # Ensure the metric exists before checking to avoid crashing early
                if self.monitor in trainer.callback_metrics:
                    self._run_early_stopping_check(trainer)
