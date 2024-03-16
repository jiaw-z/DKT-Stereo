CUDA_VISIBLE_DEVICES=0 python tools/evaluate_stereo.py \
--config "configs/raft_stereo/base.json" \
--valid_iters 32 \
--restore_ckpt 'ckpt/dkt-raft/booster_ft.pth' \
--logdir 'output/eval/dkt-raft'