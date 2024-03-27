gpus=$1
workspace=$2

CUDA_VISIBLE_DEVICES=$1 python tools/ft_dkt.py --train_datasets booster \
--config configs/raft_stereo/base.json \
--batch_size 2 --num_steps 5000 --image_size 480 896 --lr 1e-5 \
--ema_decay 0.9999  --tau_pl 3.0 \
--save_dir $workspace/stage1 \
--restore_ckpt model_zoo/stereo/RAFT-Stereo/raftstereo-sceneflow.pth


CUDA_VISIBLE_DEVICES=$1 python tools/ft_dkt.py --train_datasets booster \
--config configs/raft_stereo/base.json \
--batch_size 2 --num_steps 5000 --image_size 480 896 --lr 1e-5 \
--ema_decay 0.99999  --tau_pl 3.0 \
--save_dir $workspace/stage2 \
--restore_ckpt $workspace/stage1/5000_model.pth \
--restore_ckpt_T model_zoo/stereo/RAFT-Stereo/raftstereo-sceneflow.pth