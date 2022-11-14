## Masked Autoencoder and DINO image reconstruction

## Setup
- Install [conda](https://docs.conda.io/en/latest/miniconda.html).
- Create conda environment by running `conda env create`.

## Training

To load the frozen encoder weights from DINO and train the image decoder:

```
python -m torch.distributed.launch --nproc_per_node=4 main_pretrain.py  --batch_size 64 \
    --world_size 4 \
    --accum_iter 16 \
    --model mae_vit_small_patch16 \
    --mask_ratio 0.0 \
    --epochs 200 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05  \
    --output_dir /mnt/remote/data/users/debug \
    --data_path /mnt/remote/data/users/anthony/dino_and_mae/dino_train \
    --resume /mnt/remote/data/users/anthony/dino_and_mae/dino_encoder_weights.pth \
    --pretrained_model dino
```

To load the frozen encoder weights from Masked Autoencoder and train the image decoder:

```
python -m torch.distributed.launch --nproc_per_node=8 main_pretrain.py  --batch_size 64 \
    --world_size 8 \
    --accum_iter 8 \
    --model mae_vit_base_patch16 \
    --mask_ratio 0.0 \
    --epochs 200 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05  \
    --output_dir /mnt/remote/data/users/debug \
    --data_path /mnt/remote/data/users/anthony/dino_and_mae/dino_train \
    --resume /mnt/remote/data/users/anthony/dino_and_mae/mae_encoder_weights.pth \
    --pretrained_model mae
```