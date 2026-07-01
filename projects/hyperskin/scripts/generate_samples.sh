# FastGAN Melanoma
WANDB_MODE=disabled python src/main.py predict -c logs/hypersynth/j7od0bbk/config.yaml --ckpt_path="logs/hypersynth/j7od0bbk/checkpoints/step=0-val_MIFID=114.7889.ckpt" --trainer.logger=false

# FastGAN Nevi
WANDB_MODE=disabled python src/main.py predict -c logs/hypersynth/3113cnnm/config.yaml --ckpt_path="logs/hypersynth/3113cnnm/checkpoints/step=0-val_FID=99.1142.ckpt" --trainer.logger=false

# SPADE FastGAN Melanoma
WANDB_MODE=disabled python src/main.py predict -c logs/hypersynth/16ztzy8j/config.yaml --ckpt_path="logs/hypersynth/16ztzy8j/checkpoints/step=0-val_FID=86.3025.ckpt" --trainer.logger=false

# CycleGAN Melanoma
WANDB_MODE=disabled python src/main.py predict -c logs/hypersynth/i3nz2pqi/config.yaml --ckpt_path="logs/hypersynth/i3nz2pqi/checkpoints/step=0-val_FID=104.0855.ckpt" --trainer.logger=false

# AC CycleGAN Melanoma + Nevi
WANDB_MODE=disabled python src/main.py predict -c logs/hypersynth/ju3w3sw1/config.yaml --ckpt_path="logs/hypersynth/ju3w3sw1/checkpoints/step=0-val_FID=117.5433.ckpt" --trainer.logger=false

# FastGAN Trained with Melanoma + Nevi
WANDB_MODE=disabled python src/main.py predict -c logs/hypersynth/98xb02br/config.yaml --ckpt_path="logs/hypersynth/98xb02br/checkpoints/step=0-val_FID=80.2161.ckpt" --trainer.logger=false

# FastGAN Pretrained Melanoma
WANDB_MODE=disabled python src/main.py predict -c logs/hypersynth/0ta2r1jy/config.yaml --ckpt_path="logs/hypersynth/0ta2r1jy/checkpoints/step=0-val_FID=90.5197.ckpt" --trainer.logger=false

# FastGAN Pretrained Nevi
WANDB_MODE=disabled python src/main.py predict -c logs/hypersynth/1r91ijoj/config.yaml --ckpt_path="logs/hypersynth/1r91ijoj/checkpoints/step=0-val_FID=97.4476.ckpt" --trainer.logger=false


WANDB_MODE=disabled python src/main.py predict -c logs/hypersynth/i3nz2pqi/config_no-pred.yaml --ckpt_path logs/hypersynth/i3nz2pqi/checkpoints/cyclegan_i3nz2pqi.ckpt --trainer.logger=false --data.init_args.rgb_only=true --model.init_args.pred_hyperspectral=true  --data.init_args.pred_num_samples=500

#cyclegan new
WANDB_MODE=disabled python src/main.py predict -c logs/hypersynth/i3nz2pqi/config_no-pred.yaml --ckpt_path logs/hypersynth/i3nz2pqi/checkpoints/cyclegan_i3nz2pqi.ckpt --trainer.logger=false --data.init_args.rgb_only=true --model.init_args.pred_hyperspectral=true  --data.init_args.pred_num_samples=500

#fastgan new melanoma: 
WANDB_MODE=disabled python src/main.py predict -c logs/hypersynth/fastgan_melanoma/config_melanoma.yaml --ckpt_path="logs/hypersynth/fastgan_melanoma/checkpoints/fastgan_melanoma_step=0-val_MIFID=114.7889.ckpt" --trainer.logger=false --data.init_args.pred_num_samples=500

#fastgan nevi
WANDB_MODE=disabled python src/main.py predict -c logs/hypersynth/fastgan_nevi/config.yaml --ckpt_path="logs/hypersynth/fastgan_nevi/checkpoints/step=0-val_FID=99.1142.ckpt" --trainer.logger=false --data.init_args.pred_num_samples=500

#cyclegan complete dataset
WANDB_MODE=disabled python src/main.py predict -c logs/hypersynth/i3nz2pqi/config_no-pred_complete.yaml --ckpt_path logs/hypersynth/i3nz2pqi/checkpoints/cyclegan_i3nz2pqi.ckpt --trainer.logger=false --data.init_args.rgb_only=true --model.init_args.pred_hyperspectral=true  --data.init_args.pred_num_samples=784