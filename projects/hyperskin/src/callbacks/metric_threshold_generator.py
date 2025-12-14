import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from torchvision.utils import save_image
import math
import wandb
from scipy.io import savemat

class MetricThresholdGenerationCallback(Callback):
    def __init__(
        self,
        monitor: str,
        thresholds: list[float],
        num_samples: int = 100,
        mode: str = "min",
        output_dir: str = "threshold_samples",
        verbose: bool = True,
    ):
        """
        Args:
            monitor: The metric name to monitor (e.g., 'val/FID').
            thresholds: List of values. Triggers generation when metric crosses these.
            num_samples: Number of images to generate and save.
            mode: 'min' (lower is better) or 'max' (higher is better).
            output_dir: Root directory to save samples.
            verbose: Print status updates.
        """
        super().__init__()
        self.monitor = monitor
        self.num_samples = num_samples
        self.mode = mode
        self.output_dir = output_dir
        self.verbose = verbose

        # Sort thresholds based on mode so we can check them in order
        # min mode: check descending (e.g. 400 -> 300)
        # max mode: check ascending (e.g. 10 -> 20)
        self.thresholds = sorted(thresholds, reverse=(mode == "min"))

        # Keep track of which thresholds we have already passed
        self.triggered_thresholds = set()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """
        Check metrics at the end of training batch (where your custom validation runs).
        """
        # 1. Check if metric exists in current logs
        metrics = trainer.callback_metrics
        if self.monitor not in metrics:
            return

        current_val = metrics[self.monitor].item()

        # 2. Identify newly crossed thresholds
        thresholds_to_trigger = []

        for t in self.thresholds:
            if t in self.triggered_thresholds:
                continue

            # Check condition
            passed = False
            if self.mode == "min" and current_val <= t:
                passed = True
            elif self.mode == "max" and current_val >= t:
                passed = True

            if passed:
                thresholds_to_trigger.append(t)

        # 3. If any threshold passed, generate samples once
        if thresholds_to_trigger:
            if self.verbose:
                print(
                    f"[Callback] Metric {self.monitor} ({current_val:.4f}) crossed thresholds: {thresholds_to_trigger}"
                )

            # Mark as triggered so we don't save again for these specific values
            self.triggered_thresholds.update(thresholds_to_trigger)

            # Generate and Save
            self._generate_and_save(trainer, pl_module, current_val)

    def _generate_and_save(self, trainer, pl_module, metric_value):
        """
        Handles the EMA swapping, data loading, generation, and saving.
        """
        # --- 1. Setup Folder Name ---
        step = trainer.global_step

        # ... (Same folder setup code as before) ...
        wandb_id = "norun"
        if isinstance(trainer.logger, pl.loggers.WandbLogger):
            wandb_id = trainer.logger.experiment.id
        elif wandb.run is not None:
            wandb_id = wandb.run.id

        metric_name_clean = self.monitor.replace("/", "_")
        folder_name = f"step={step}_{metric_name_clean}={metric_value:.4f}_{wandb_id}"

        if wandb_id != "norun":
            project = wandb.run.project
            base_dir = f"logs/{project}/{wandb_id}"
            save_path = os.path.join(base_dir, self.output_dir, folder_name)
        else:
            save_path = os.path.join(self.output_dir, folder_name)
        os.makedirs(save_path, exist_ok=True)

        # --- 2. EMA Context Manager ---

        # Identify the specific generator network to toggle
        # We avoid toggling the whole pl_module to protect Inception/FID modules
        target_net = None
        if hasattr(pl_module, "netG"):
            target_net = pl_module.netG
        elif hasattr(pl_module, "G_AB"):
            target_net = pl_module.G_AB
        else:
            # Fallback: toggle module but we must fix Inception later
            target_net = pl_module

        # Save current mode
        was_training = target_net.training

        # Swap weights
        backup_params = self._swap_ema_weights(pl_module, to_avg=True)

        # Set ONLY the generator to eval
        target_net.eval()

        try:
            self._generate_loop(trainer, pl_module, save_path)
        finally:
            # Restore original weights 
            self._swap_ema_weights(pl_module, to_avg=False, backup_params=backup_params)

            # Restore training mode ONLY for the generator
            target_net.train(was_training)

            # SAFEGUARD: If we fell back to toggling pl_module, or just to be safe,
            # force metric models back to eval
            if hasattr(pl_module, "inception_model"):
                pl_module.inception_model.eval()
            if hasattr(pl_module, "fid"):
                pl_module.fid.eval()
            if hasattr(pl_module, "classifier") and pl_module.classifier is not None:
                pl_module.classifier.eval()

    def _generate_loop(self, trainer, pl_module, save_path):
        """
        Generates num_samples images. Handles logic for FastGAN (Noise) vs CycleGAN (Images).
        """
        device = pl_module.device
        generated_count = 0

        # Determine Generation Type
        is_cyclegan = hasattr(pl_module, "G_AB")
        is_spade = hasattr(pl_module, "hparams") and getattr(pl_module.hparams, "use_spade", False)

        # Prepare Data Iterator if input images are needed
        data_iter = None
        if is_cyclegan or is_spade:
            # Create a fresh iterator from the train dataloader
            # We use train_dataloader because provided code indicates infinite sampler on train
            data_iter = iter(trainer.train_dataloader)

        with torch.no_grad():
            while generated_count < self.num_samples:
                fake_batch = None

                # --- CASE A: CycleGAN (Needs RGB Input) ---
                if is_cyclegan:
                    batch = next(data_iter)
                    # Move to device
                    real_rgb = batch["rgb"]["image"].to(device)
                    # Generate
                    fake_batch = pl_module.G_AB(real_rgb)

                # --- CASE B: FastGAN (Noise or SPADE) ---
                else:
                    batch_size = 8  # Default small batch for generation

                    if is_spade:
                        batch = next(data_iter)

                        # Handle DataModule structure (JointRGBHSI vs others)
                        if "rgb" in batch:
                            input_batch = batch["rgb"]
                        else:
                            input_batch = batch

                        # Extract conditioning
                        seg = None
                        if pl_module.hparams.spade_conditioning == "rgb_mask":
                            seg = input_batch["mask"]
                            if seg.ndim == 3:
                                seg = seg.unsqueeze(1)
                        elif pl_module.hparams.spade_conditioning == "rgb_image":
                            seg = input_batch["image"]

                        seg = seg.to(device)
                        batch_size = seg.size(0)

                        noise = torch.randn(batch_size, pl_module.hparams.nz, device=device)
                        fake_batch = pl_module(noise, seg)[0]

                    else:
                        # Unconditional FastGAN
                        noise = torch.randn(batch_size, pl_module.hparams.nz, device=device)
                        fake_batch = pl_module(noise)[0]

                # --- Post-Processing & Saving ---
                # Normalize from [-1, 1] to [0, 1]
                fake_batch = (fake_batch + 1) / 2.0
                fake_batch = fake_batch.clamp(0, 1)

                # If Hyperspectral (C > 3), take mean or select bands for visualization
                # if fake_batch.size(1) > 3:
                #     # Simple visualization strategy: average to grayscale then repeat to RGB
                #     fake_batch = fake_batch.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)

                # Save individual images
                for i in range(fake_batch.size(0)):
                    if generated_count >= self.num_samples:
                        break

                    file_name = f"sample_{generated_count:05d}.mat"
                    # save_image(fake_batch[i], os.path.join(save_path, file_name))
                    img_array = fake_batch[i].cpu().permute(1, 2, 0).numpy()  # HWC format
                    savemat(os.path.join(save_path, file_name), {'image': img_array})
                    generated_count += 1

        if self.verbose:
            print(f"[Callback] Saved {generated_count} samples to {save_path}")

    def _swap_ema_weights(self, pl_module, to_avg: bool, backup_params=None):
        """
        Generic helper to handle EMA swapping for both FastGAN and CycleGAN.
        Detects which lists of parameters to swap based on module attributes.
        """
        # Identify the networks and the stored EMA params
        networks = []
        avg_param_storage = getattr(pl_module, "avg_param_G", None)

        if avg_param_storage is None:
            return None  # No EMA usage detected

        if hasattr(pl_module, "G_AB") and hasattr(pl_module, "G_BA"):
            # CycleGAN Case
            networks = list(pl_module.G_AB.parameters()) + list(pl_module.G_BA.parameters())
        elif hasattr(pl_module, "netG"):
            # FastGAN Case
            networks = list(pl_module.netG.parameters())
        else:
            return None  # Unknown model structure

        if to_avg:
            # Backup current current weights
            current_backup = [p.data.clone().detach() for p in networks]

            # Load average weights into model
            for p, avg_p in zip(networks, avg_param_storage):
                p.data.copy_(avg_p)

            return current_backup
        else:
            # Restore backup (standard training weights)
            if backup_params is not None:
                for p, backup_p in zip(networks, backup_params):
                    p.data.copy_(backup_p)
            return None
