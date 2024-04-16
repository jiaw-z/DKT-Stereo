# DKT-Stereo: Robust Synthetic-to-Real Transfer for Stereo Matching
This is the official repository for our CVPR 2024 paper: Robust Synthetic-to-Real Transfer for Stereo Matching.

paper: [[arxiv](https://arxiv.org/pdf/2403.07705)]

## Introduction
We aim to fine-tune stereo networks without compromising robustness to unseen domains. We identify that learning new knowledge without sufficient regularization and overfitting GT details can degrade the robustness. We propose the DKT framework, which improves fine-tuning by dynamically measuring what has been learned.


<p align="center">
  <img src="https://github.com/jiaw-z/DKT-Stereo/assets/66359549/9898679f-60c6-4624-92b4-4874a1ba3b53" />
</p>

![image](https://github.com/jiaw-z/DKT-Stereo/assets/66359549/3b115d58-f441-4d56-9bf4-67bf87b28ad6)




## TODO
- [x] Release Training Code.
- [x] Release Checkpoint.

## Demos
Fine-tuned checkpoints of DKT-Stereo can be downloaded from [google drive](https://drive.google.com/drive/folders/1EtBp8biVF21rYCc_gJHCW2sUkowWMPcR?usp=sharing)

The sceneflow pre-trained checkpoints can be obtained from [IGEV](https://github.com/gangweiX/IGEV) and [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo).

## Environment

* NVIDIA RTX 3090
* Python 3.8
* pytorch 1.12

  ### Create a virtual environment and activate it.

```
conda create -n DKT_Stereo python=3.8
conda activate DKT_Stereo
```
### Dependencies

```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -c nvidia
pip install opencv-python
pip install scikit-image
pip install tensorboard
pip install matplotlib 
pip install tqdm
pip install timm==0.5.4
```

## Required Data

To evaluate/train DKT-Stereo, you will need to download the required datasets. 
* [KITTI](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)
* [Booster](https://cvlab-unibo.github.io/booster-web/)
* [Middlebury](https://vision.middlebury.edu/stereo/submit3/)
* [ETH3D](https://www.eth3d.net/datasets#low-res-two-view-test-data)

By default `stereo_datasets.py` will search for the datasets in these locations. 

```
├── /data
    ├── KITTI
        ├── KITTI_2012
            ├── training
            ├── testing
        ├── KITTI_2015
            ├── training
            ├── testing
    ├── Booster_dataset
        ├── full
        ├── half
        ├── quarter
            ├── train
    ├── Middlebury
        ├── MiddEval3
            ├── trainingF
            ├── trainingH
            ├── trainingQ
    ├── ETH3D
        ├── two_view_training
        ├── two_view_training_gt

```

## Evaluation
```Shell
python tools/evaluate_stereo.py --config configs/raft_stereo/base.json --restore_ckpt ckpt/dkt-raft/booster_ft.pth --logdir output/eval/dkt-raft
```
```Shell
python tools/evaluate_stereo.py --config configs/igev_stereo/base.json --restore_ckpt ckpt/dkt-igev/kitti_ft.pth --logdir output/eval/dkt-igev
```
## Training
Booster fine-tuning. This current fine-tuning code on booster is different from the implementation for online submission checkpoints, which use the cascade training strategy as [PCVNet](https://github.com/jiaxiZeng/Parameterized-Cost-Volume-for-Stereo-Matching).
```Shell
bash run_scripts/raft-stereo/ft_booster.sh gpus(0,1) output_dir(/output/raftstereo/booster_ft)
```

KITTI fine-tuning.
```Shell
bash run_scripts/igev/ft_kitti.sh gpus(0,1,2,3) output_dir(/output/igevstereo/kitti_ft)
```


## Citation

If you find our work useful in your research, please consider citing our paper:

```bibtex
@article{zhang2024robust,
  title={Robust Synthetic-to-Real Transfer for Stereo Matching},
  author={Zhang, Jiawei and Li, Jiahe and Huang, Lei and Yu, Xiaohan and Gu, Lin and Zheng, Jin and Bai, Xiao},
  journal={arXiv preprint arXiv:2403.07705},
  year={2024}
}
```


# Acknowledgements

This project is based on [IGEV](https://github.com/gangweiX/IGEV) and [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo). Thanks for these great projects!
