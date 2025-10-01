import torch
import src.data_modules.datasets.hsi_dermoscopy_dataset
torch.serialization.add_safe_globals([src.data_modules.datasets.hsi_dermoscopy_dataset.HSIDermoscopyTask])

checkpoint = torch.load("logs/hypersynth/omrvye06/checkpoints/epoch=02-val_acc=0.2370.ckpt")
print(checkpoint["hyper_parameters"])