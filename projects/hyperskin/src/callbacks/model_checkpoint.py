from pytorch_lightning.callbacks import ModelCheckpoint
import torch

class FixedModelCheckpoint(ModelCheckpoint):
    def _save_checkpoint(self, trainer, filepath: str) -> None:
        """Override save to fix '_class_path' -> 'class_path' in hyper_parameters before writing."""
        # First, let Trainer create the checkpoint dict (not yet written to disk)
        checkpoint = trainer._checkpoint_connector.dump_checkpoint(weights_only=self.save_weights_only)

        # Fix the hyper_parameters section if necessary
        if "hyper_parameters" in checkpoint:
            hparams = checkpoint["hyper_parameters"]
            if "_class_path" in hparams:
                hparams["class_path"] = hparams.pop("_class_path")

        # Now actually save it using torch
        trainer.save_checkpoint(filepath, weights_only=self.save_weights_only)

        # ⚠️ The above line by default recreates the checkpoint dict (!!)
        # so we need to manually save the modified one instead:
        torch.save(checkpoint, filepath)

        # Update bookkeeping
        self._last_global_step_saved = trainer.global_step
        self._last_checkpoint_saved = filepath

        # Notify loggers as usual
        if trainer.is_global_zero:
            for logger in trainer.loggers:
                logger.after_save_checkpoint(self)
