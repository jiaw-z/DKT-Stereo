gpus=$1
workspace=$2

CUDA_VISIBLE_DEVICES=$1 python tools/ft_dkt.py --train_datasets kitti_mix \
--config configs/igev_stereo/base.json \
--batch_size 4 --num_steps 5000 --image_size 320 736 --lr 2e-4 \
--ema_decay 0.99  --tau_pl 3.0 \
--save_dir $workspace/stage1 \
--restore_ckpt model_zoo/stereo/IGEV-Stereo/sceneflow.pth


CUDA_VISIBLE_DEVICES=$1 python tools/ft_dkt.py --train_datasets kitti_mix \
--config configs/igev_stereo/base.json \
--batch_size 4 --num_steps 50000 --image_size 320 736 --lr 2e-4 \
--ema_decay 0.99999  --tau_pl 0.5 \
--save_dir $workspace/stage2 \
--restore_ckpt $workspace/stage1/5000_model.pth \
--restore_ckpt_T model_zoo/stereo/IGEV-Stereo/sceneflow.pth