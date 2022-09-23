## Pre-training MAE
To train on a single node:
```
python -m torch.distributed.launch --nproc_per_node=8 main_pretrain.py  --batch_size 64 \
    --world_size 8 \
    --accum_iter 8 \
    --model mae_vit_base_patch16 \
    --mask_ratio 0.75 \
    --epochs 800 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05  \
    --output_dir /home/wayve/anthony/experiments/mae/vit_base \
    --data_path /mnt/local/datasets/dino_train
```

To train mae decoder on branch `image-reconstruction`
```
python -m torch.distributed.launch --nproc_per_node=8 main_pretrain.py  --batch_size 64 \
    --world_size 8 \
    --accum_iter 8 \
    --model mae_vit_base_patch16 \
    --mask_ratio 0.0 \
    --epochs 200 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05  \
    --output_dir /home/wayve/anthony/experiments/mae/mae_decoder \
    --data_path /mnt/local/datasets/dino_train \
    --resume /home/wayve/anthony/experiments/mae/vit_base_ipace/checkpoint-180.pth \
    --pretrained_model mae
```

To train dino decoder on branch `image-reconstruction`
```
python -m torch.distributed.launch --nproc_per_node=4 main_pretrain.py  --batch_size 64 \
    --world_size 4 \
    --accum_iter 16 \
    --model mae_vit_small_patch16 \
    --mask_ratio 0.0 \
    --epochs 200 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05  \
    --output_dir /home/wayve/anthony/experiments/mae/dino_decoder \
    --data_path /mnt/local/datasets/dino_train \
    --resume /home/wayve/anthony/experiments/dino/ipace_data/checkpoint0080.pth \
    --pretrained_model dino
```


To pre-train ViT-Large (recommended default) with **multi-node distributed training**, run the following on 8 nodes with 8 GPUs each:
```
python submitit_pretrain.py \
    --job_dir ${JOB_DIR} \
    --nodes 8 \
    --use_volta32 \
    --batch_size 64 \
    --model mae_vit_large_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 800 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path ${IMAGENET_DIR}
```
- Here the effective batch size is 64 (`batch_size` per gpu) * 8 (`nodes`) * 8 (gpus per node) = 4096. If memory or # gpus is limited, use `--accum_iter` to maintain the effective batch size, which is `batch_size` (per gpu) * `nodes` * 8 (gpus per node) * `accum_iter`.
- `blr` is the base learning rate. The actual `lr` is computed by the [linear scaling rule](https://arxiv.org/abs/1706.02677): `lr` = `blr` * effective batch size / 256.
- Here we use `--norm_pix_loss` as the target for better representation learning. To train a baseline model (e.g., for visualization), use pixel-based construction and turn off `--norm_pix_loss`.
- The exact same hyper-parameters and configs (initialization, augmentation, etc.) are used as our TF/TPU implementation. In our sanity checks, this PT/GPU re-implementation can reproduce the TF/TPU results within reasonable random variation. We get 85.5% [fine-tuning](FINETUNE.md) accuracy by pre-training ViT-Large for 800 epochs (85.4% in paper Table 1d with TF/TPU).
- Training time is ~42h in 64 V100 GPUs (800 epochs).

To train ViT-Base or ViT-Huge, set `--model mae_vit_base_patch16` or `--model mae_vit_huge_patch14`.
