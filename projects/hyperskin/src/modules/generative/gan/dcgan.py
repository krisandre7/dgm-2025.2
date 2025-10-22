"""
DCGAN (Deep Convolutional Generative Adversarial Networks) Architecture:

DCGANs are a type of GAN that use convolutional and convolutional-transpose layers
in the discriminator and generator, respectively. They introduced several architectural
guidelines to stabilize the training of GANs, resulting in higher quality image generation.

Reference: Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks.
- https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
"""

import torch
import torch.nn as nn
import torch.nn.utils as utils
from torch import Tensor
from torch.nn.functional import binary_cross_entropy_with_logits as bce_with_logits
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance

from src.modules.generative.gan.gan import GAN


def initialize_weights(model: nn.Module):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    return model


class Generator(nn.Module):
    def __init__(
        self,
        img_size: int,
        img_channels: int,
        latent_dim: int,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        if img_size == 64:
            self.model = nn.Sequential(
                self._block(latent_dim, 1024, 4, 1, 0),
                self._block(1024, 512, 4, 2, 1),
                self._block(512, 256, 4, 2, 1),
                self._block(256, 128, 4, 2, 1),
                self._block(128, img_channels, 4, 2, 1, final_layer=True),
            )

        elif img_size == 28:
            # MNIST
            self.model = nn.Sequential(
                self._block(latent_dim, 256, 7, 1, 0),
                self._block(256, 128, 4, 2, 1),
                self._block(128, img_channels, 4, 2, 1, final_layer=True),
            )
        elif img_size == 256:
            # deeper architecture for 256x256
            self.model = nn.Sequential(
                self._block(latent_dim, 2048, 4, 1, 0),
                self._block(2048, 1024, 4, 2, 1),
                self._block(1024, 512, 4, 2, 1),
                self._block(512, 256, 4, 2, 1),
                self._block(256, 128, 4, 2, 1),
                self._block(128, 64, 4, 2, 1),
                self._block(64, img_channels, 4, 2, 1, final_layer=True),
            )
        else:
            raise ValueError("Image size not supported.")

        self.model = initialize_weights(self.model)

    @staticmethod
    def _block(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        final_layer: bool = False,
    ) -> nn.Sequential:
        """
        Returns a block for the generator containing
            a fractional-strided convolution,
            batch normalization,
            and a ReLU activation for all layers except for the output, which uses the Tanh activation.
        """
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels) if not final_layer else nn.Identity(),
            nn.ReLU(inplace=True) if not final_layer else nn.Tanh(),
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.model(z)

    def random_sample(self, batch_size: int) -> Tensor:
        z = torch.randn(
            [batch_size, self.latent_dim, 1, 1],
            device=self.device,
        )
        return self(z)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


# class Discriminator(nn.Module):
#     def __init__(
#         self,
#         img_size: int,
#         img_channels: int,
#     ) -> None:
#         super().__init__()

#         if img_size == 64:
#             self.model = nn.Sequential(
#                 self._block(img_channels, 64, 4, 2, 1, use_bn=False),
#                 self._block(64, 128, 4, 2, 1, use_bn=True),
#                 self._block(128, 256, 4, 2, 1, use_bn=True),
#                 self._block(256, 512, 4, 2, 1, use_bn=True),
#                 self._block(512, 1, 4, 1, 0, use_bn=False, final_layer=True),
#             )

#         elif img_size == 128:
#             self.model = nn.Sequential(
#                 self._block(img_channels, 64, 4, 2, 1, use_bn=False),
#                 self._block(64, 128, 4, 2, 1, use_bn=True),
#                 self._block(128, 256, 7, 1, 0),
#                 self._block(256, 1, 1, 1, 0, use_bn=False, final_layer=True),
#             )
#         elif img_size == 256:
#             self.model = nn.Sequential(
#                 self._block(img_channels, 64, 4, 2, 1, use_bn=False),
#                 self._block(64, 128, 4, 2, 1),
#                 self._block(128, 256, 4, 2, 1),
#                 self._block(256, 512, 4, 2, 1),
#                 self._block(512, 1024, 4, 2, 1),
#                 self._block(1024, 1, 4, 1, 0, use_bn=False, final_layer=True),
#             )
#         else:
#             raise ValueError("Image size not supported.")

#         self.model = initialize_weights(self.model)

#     @staticmethod
#     def _block(
#         in_channels: int,
#         out_channels: int,
#         kernel_size: int,
#         stride: int,
#         padding: int,
#         use_bn: bool = True,
#         final_layer: bool = False,
#     ) -> nn.Sequential:
#         """
#         Returns a block for the discriminator containing
#         - a strided convolution,
#         - optional batch normalization,
#         - and a LeakyReLU activation.
#         """
#         return nn.Sequential(
#             nn.Conv2d(
#                 in_channels,
#                 out_channels,
#                 kernel_size,
#                 stride,
#                 padding,
#                 bias=False,
#             ),
#             nn.BatchNorm2d(out_channels) if use_bn else nn.Identity(),
#             nn.LeakyReLU(0.2, inplace=True) if not final_layer else nn.Identity(),
#         )

#     def forward(self, x: Tensor) -> Tensor:
#         return self.model(x).squeeze()


# class Discriminator(nn.Module):
#     """
#     3D convolutional discriminator for hyperspectral images.
#     Convolves across both spatial (H, W) and spectral (bands) dimensions.
#     """
#     def __init__(self, img_channels=16, img_size=64):
#         super().__init__()

#         # Expect input [B, 1, C, H, W] where C=img_channels
#         self.model = nn.Sequential(
#             # (in_channels, out_channels, kernel_size, stride, padding)
#             nn.Conv3d(1, 32, (3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),  # spectral conv
#             nn.LeakyReLU(0.2, inplace=True),

#             nn.Conv3d(32, 64, (3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
#             nn.BatchNorm3d(64),
#             nn.LeakyReLU(0.2, inplace=True),

#             nn.Conv3d(64, 128, (3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),  # reduces spectral depth too
#             nn.BatchNorm3d(128),
#             nn.LeakyReLU(0.2, inplace=True),

#             nn.Conv3d(128, 256, (3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
#             nn.BatchNorm3d(256),
#             nn.LeakyReLU(0.2, inplace=True),

#             nn.Conv3d(256, 1, (2, 4, 4), stride=(1, 1, 1), padding=0),  # final score
#         )

#     def forward(self, x):
#         # input: [B, C, H, W]
#         x = x.unsqueeze(1)  # → [B, 1, C, H, W]
#         out = self.model(x)
#         return out.view(x.size(0), -1)  # flatten to [B, 1]



# class Discriminator(nn.Module):
#     """
#     3D convolutional discriminator for hyperspectral images.
#     Convolves across both spatial (H, W) and spectral (bands) dimensions.
#     Uses Spectral Normalization for Lipschitz constraint.
#     """
#     def __init__(self, img_channels=16, img_size=64):
#         super().__init__()

#         self.model = nn.Sequential(
#             # (in_channels, out_channels, kernel_size, stride, padding)
#             utils.spectral_norm(nn.Conv3d(1, 32, (3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1))),
#             nn.LeakyReLU(0.2, inplace=True),

#             utils.spectral_norm(nn.Conv3d(32, 64, (3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1))),
#             # nn.BatchNorm3d(64),
#             nn.LeakyReLU(0.2, inplace=True),

#             utils.spectral_norm(nn.Conv3d(64, 128, (3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))),
#             # nn.BatchNorm3d(128),
#             nn.LeakyReLU(0.2, inplace=True),

#             utils.spectral_norm(nn.Conv3d(128, 256, (3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))),
#             # nn.BatchNorm3d(256),
#             nn.LeakyReLU(0.2, inplace=True),

#             utils.spectral_norm(nn.Conv3d(256, 1, (2, 4, 4), stride=(1, 1, 1), padding=0))
#         )

#     def forward(self, x):
#         # input: [B, C, H, W]
#         x = x.unsqueeze(1)  # → [B, 1, C, H, W]
#         out = self.model(x)
#         return out.view(x.size(0), -1)  # flatten to [B, 1]


class Discriminator(nn.Module):
    """
    Dual-branch 3D Discriminator following the SHS-GAN architecture.
    
    - Arm 1: spatial–spectral 3D convolutional branch (original HS cube).
    - Arm 2: frequency branch operating on the spectral FFT of the cube.
    
    When fft_arm=False, only the spatial–spectral branch is used.
    """

    def __init__(self, img_channels=16, img_size=64, use_fft_mag=True, fft_arm=True):
        super().__init__()
        self.use_fft_mag = use_fft_mag
        self.fft_arm = fft_arm

        # --------------------------
        # Helper convolutional block
        # --------------------------
        def block(in_c, out_c, stride=(1, 2, 2)):
            return nn.Sequential(
                utils.spectral_norm(
                    nn.Conv3d(in_c, out_c, (3, 4, 4), stride=stride, padding=(1, 1, 1))
                ),
                nn.LeakyReLU(0.2, inplace=True)
            )


        self.spatial_branch = nn.Sequential(
            block(1, 32),
            block(32, 64),
            block(64, 128, stride=(2, 2, 2)),
            block(128, 256, stride=(2, 2, 2))
        )

        if fft_arm:
            in_fft_channels = 1 if use_fft_mag else 2
            self.freq_branch = nn.Sequential(
                block(in_fft_channels, 32),
                block(32, 64),
                block(64, 128, stride=(2, 2, 2)),
                block(128, 256, stride=(2, 2, 2))
            )

            # Fusion + classifier (for dual-arm setup)
            self.classifier = nn.Sequential(
                utils.spectral_norm(nn.Conv3d(512, 256, (2, 4, 4), stride=1, padding=0)),
                nn.LeakyReLU(0.2, inplace=True),
                utils.spectral_norm(nn.Conv3d(256, 1, 1))
            )
        else:
            # Single-arm classifier (only spatial)
            self.classifier = nn.Sequential(
                utils.spectral_norm(nn.Conv3d(256, 128, (2, 4, 4), stride=1, padding=0)),
                nn.LeakyReLU(0.2, inplace=True),
                utils.spectral_norm(nn.Conv3d(128, 1, 1))
            )

    # --------------------------
    # Forward Pass
    # --------------------------
    def forward(self, x):
        """
        x: [B, C, H, W] where
            C = spectral bands,
            H, W = spatial dimensions.
        """
        x = x.unsqueeze(1)  # → [B, 1, C, H, W]

        # --- Spatial–Spectral Path ---
        feat_spatial = self.spatial_branch(x)

        # If FFT arm disabled, only use spatial path
        if not self.fft_arm:
            out = self.classifier(feat_spatial)
            return out.view(x.size(0), -1)

        # --- Frequency Path (FFT over spectral dimension) ---
        # Apply 1D FFT along the spectral (band) axis: dim=2
        fft = torch.fft.fft(x, dim=2, norm="ortho")

        if self.use_fft_mag:
            fft_input = fft.abs()  # use magnitude of spectral FFT
        else:
            fft_input = torch.cat((fft.real, fft.imag), dim=1)  # [B, 2, C, H, W]

        feat_freq = self.freq_branch(fft_input)

        # --- Fusion ---
        fused = torch.cat([feat_spatial, feat_freq], dim=1)  # concat along channel dim

        # --- Final decision ---
        out = self.classifier(fused)
        return out.view(x.size(0), -1)

class DCGAN(GAN):
    """
    Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks
    In recent years, supervised learning with convolutional networks (CNNs) has seen huge adoption in computer vision \
        applications.
    Comparatively, unsupervised learning with CNNs has received less attention.
    In this work we hope to help bridge the gap between the success of CNNs for supervised learning and unsupervised
        learning.
    We introduce a class of CNNs called deep convolutional generative adversarial networks (DCGANs), that have certain
        architectural constraints, and demonstrate that they are a strong candidate for unsupervised learning.
    Training on various image datasets, we show convincing evidence that our deep convolutional adversarial pair learns
        a hierarchy of representations from object parts to scenes in both the generator and discriminator.
    Additionally, we use the learned features for novel tasks - demonstrating their applicability as general image
        representations.
    """

    def __init__(
        self,
        img_channels: int,
        img_size: int,
        latent_dim: int,
        d_lr: float,
        g_lr: float,
        b1: float,
        b2: float,
        weight_decay: float,
        calculate_metrics: bool = False,
        metrics: list[str] = [],
        summary: bool = True,
        log_images_after_n_epochs: int = 1,
        log_metrics_after_n_epochs: int = 1,
    ) -> None:
        super().__init__(
            img_channels=img_channels,
            img_size=img_size,
            latent_dim=latent_dim,
            d_lr=d_lr,
            g_lr=g_lr,
            b1=b1,
            b2=b2,
            weight_decay=weight_decay,
            calculate_metrics=calculate_metrics,
            metrics=metrics,
            summary=False,
            log_images_after_n_epochs=log_images_after_n_epochs,
            log_metrics_after_n_epochs=log_metrics_after_n_epochs,
        )

        self.G = Generator(
            img_size=img_size,
            img_channels=img_channels,
            latent_dim=latent_dim,
        )
        self.D = Discriminator(
            img_size=img_size,
            img_channels=img_channels,
        )

        if self.metrics:
            self.fid = FrechetInceptionDistance() if "fid" in self.metrics else None
            self.kid = KernelInceptionDistance(subset_size=100) if "kid" in self.metrics else None
            self.inception_score = InceptionScore() if "is" in self.metrics else None

        self.z = torch.randn([16, latent_dim, 1, 1])
        if summary:
            self.summary()

    def _calculate_d_loss(self, x: Tensor, x_hat: Tensor) -> Tensor:
        logits_real = self.D(x)
        d_loss_real = bce_with_logits(logits_real, torch.ones_like(logits_real))

        logits_fake = self.D(x_hat.detach())
        d_loss_fake = bce_with_logits(logits_fake, torch.zeros_like(logits_fake))

        d_loss = (d_loss_real + d_loss_fake) / 2

        loss_dict = {
            "d_loss": d_loss,
            "d_loss_real": d_loss_real,
            "d_loss_fake": d_loss_fake,
            "logits_real": logits_real.mean(),
            "logits_fake": logits_fake.mean(),
        }
        return loss_dict

    def _calculate_g_loss(self, x_hat: Tensor) -> Tensor:
        logits_fake = self.D(x_hat)
        g_loss = bce_with_logits(logits_fake, torch.ones_like(logits_fake))
        loss_dict = {"g_loss": g_loss}
        return loss_dict
