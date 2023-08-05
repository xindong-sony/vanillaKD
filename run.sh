python -m torch.distributed.launch --nproc_per_node=4 train-kd.py ~/imagenet_dataset/imagenet/ --model resnet18 --teacher beitv2_base_patch16_224 --teacher-pretrained ./weights/beitv2_base_patch16_224_pt1k_ft21kto1k_weights.pth --kd-loss kd --amp --epochs 300 --batch-size 256 --lr 5e-3 --opt lamb --sched cosine --weight-decay 0.02 --warmup-epochs 5 --warmup-lr 1e-6 --smoothing 0.0 --drop 0 --drop-path 0.05 --aug-repeats 3 --aa rand-m7-mstd0.5 --mixup 0.1 --cutmix 1.0 --color-jitter 0 --crop-pct 0.95 --bce-loss 1 --log-wandb  --use-multi-epochs-loader -j 16 --experiment shuffle_except_max --log-interval 1 --change-teacher 0