# data augmentations k-fold training script
#!/usr/bin/env bash

# no data augmentations k-fold training yes pretrained model
bash scripts/run_kfold.sh configs/model/hsi_classifier_densenet201_best.yaml configs/data/hsi_dermoscopy_croppedv2_notaug.yaml 5

#with data augmentations k-fold training yes pretrained model
bash scripts/run_kfold.sh configs/model/hsi_classifier_densenet201_best.yaml configs/data/hsi_dermoscopy_croppedv2_aug.yaml 5

# no data augmentations k-fold training yes pretrained model focalloss
bash scripts/run_kfold.sh configs/model/densenet201_best_focalloss.yaml configs/data/hsi_dermoscopy_croppedv2_notaug.yaml 5

#no data augmentations k-fold training yes pretrained synthetic data model
bash scripts/run_kfold.sh configs/model/hsi_classifier_densenet201_best.yaml configs/data/hsi_dermoscopy_croppedv2_notaugsynth.yaml 5

#with data augmentations k-fold training yes pretrained synthetic data model
bash scripts/run_kfold.sh configs/model/hsi_classifier_densenet201_best.yaml configs/data/hsi_dermoscopy_croppedv2_augsynth.yaml 5



