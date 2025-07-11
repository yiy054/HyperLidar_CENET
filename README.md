# HyperLidar: Toward Concise and Efficient LiDAR Semantic Segmentation for Autonomous Driving [![arXiv](https://img.shields.io/badge/arXiv-2207.12691-b31b1b?logo=arXiv&logoColor=green)](https://arxiv.org/abs/2207.12691)

Code for our paper:
> **CENet: Toward Concise and Efficient LiDAR Semantic Segmentation for Autonomous Driving**
> <br>Huixian Cheng, Xianfeng Han, Guoqiang Xiao<br>
> Accepted by ICME2022

## Abstract：

## Updates:
<!-- **2023-03-28[NEW:sparkles:]** CENet achieves competitive performance in robustness evaluation at SemanticKITTI. See Repo of [Robo3D](https://github.com/ldkong1205/Robo3D) for more details.
<div align="center">
  <img src="assert/robustness.png"/>
</div><br/>

**2022-07-06[:open_mouth::scream::thumbsup:]** [Ph.D. Hou](https://github.com/cardwing) reported an astounding 67.6% mIoU test performance of CENet, see [this issue](https://github.com/huixiancheng/CENet/issues/7) and [PVD Repo](https://github.com/cardwing/Codes-for-PVKD) for details. -->

## Prepare:
Download SemanticKITTI from [official web](http://www.semantic-kitti.org/dataset.html). Download SemanticPOSS from [official web](http://www.poss.pku.edu.cn./download.html).

You can use the provided [Nautilus deployment YAML](./nautilus/hyperlidar.yaml) to launch the container environment.

- The base image used is:  
  ```
  ghcr.io/darthiv02/cenet_image:1.1
  ```
  > ⚠️ This image includes only the original **CENet** backbone and does **not** include HyperLiDAR support.

- To enable **TorchHD** functionality (for HyperLiDAR), you’ll need to manually install the `torch-hd` package [torch-hd](https://github.com/hyperdimensional-computing/torchhd?tab=readme-ov-file) :
  ```bash
  pip install torch-hd
  ```

### 🔄 Switching to HyperLiDAR

To work with **HyperLiDAR**, clone the specialized repository instead:
```bash
git clone https://github.com/yiy054/HyperLidar_CENET.git
```

### 📁 Dataset Setup (on Nautilus)

If you're using the Nautilus environment and YAML deployment:

1. Download and extract the SemanticKITTI dataset:
   ```bash
   cd /mnt/data
   cp /root/main/dataset/semantickitti_fast.tar.gz .
   tar -xvzf semantickitti_fast.tar.gz
   cd /home
   ```

2. After extraction, the dataset will be available at:
   ```
   /mnt/data/semantickitti
   ```


## File Structure:
Used the 
```
.
├── LICENSE
├── README.md           // this file
├── main.py             // main file (different method...)
├── methods             // implementation of HyperLidar
├── requirements_cenet.txt
├── dataset             // dataset loader
└── config              // dataset and model config
```

## Usage：
### Train：
- Pretrain:
    <!-- ```
    sudo docker run -it --rm \              
      --runtime nvidia --gpus all \
      -v "$(pwd)/CENet:/root/CENet" \
      --name test_container \
      dustynv/l4t-pytorch:r36.2.0 \
      bash
    ``` -->
    `python train.py -d /mnt/data/semantickitti -ac config/arch/senet-512.yml -n senet-512 -l pretrain0_4 -t 0,1,2,3,4`

    `python train.py -d /mnt/data/semantickitti -ac config/arch/senet-512.yml -n senet-512 -l retrain5_10 -p pretrain0_4/senet-512 -t 5,6,7,9,10`

    `python train.py -d /mnt/data/semantickitti -ac config/arch/senet-512.yml -n senet-512 -l retrain56 -p /root/main/CENET/pretrain012347910/senet-512 -t 5,6`

    ```
    python train.py 
      -d <dataset_path> 
      -ac config/arch/senet-512.yml 
      -l <save_dic> 
      -n senet-512 
      -p <pretrain_model_path> 
      -t <train_seqs>
    ```

    Note that the following training strategy is used due to GPU and time constraints, see [kitti.sh](https://github.com/huixiancheng/SENet/blob/main/kitti.sh) for details.

    First train the model with 64x512 inputs. Then load the pre-trained model to train the model with 64x1024 inputs, and finally load the pre-trained model to train the model with 64x2048 inputs.
    
    Also, for this reason, if you want to resume training from a breakpoint, uncomment [this section](https://github.com/huixiancheng/SENet/blob/c5827853ee32660ad9487a679890822ac9bf8bf8/modules/trainer.py#L193-L203) and change "/SENet_valid_best" to "/SENet".

    Currently, HyperLidar only focuse on senet-512 size and used the retrain as the online learning process. Therefore, the train sequence will be change when the pretrain model exist and need to used the dataset sequence that never seem before. 
- Online learning:

  `python main.py -d /mnt/data/semantickitti -l your_predictions_path -m pretrain0_4/senet-512 -t 5,6,7,9,10`

  `python main.py -d /root/main/dataset/semantickitti -l ./temp_prediction -m /root/main/CENET/pretrain012347910/senet-512 -t 5,6 > ./retrain56_Conv.log 2>&1`

  python main.py -d /mnt/data/nuscenes_semantickitti -l ./temp_prediction -m /root/main/CENET/pretrain012347910/senet-512 -t 5,6

  ```
  python main.py 
    -d <dataset_path> 
    -l <save_pred_dic> 
    -m <pretrain_model_path>
    -t <train_seqs>
  ```
### CENET Infer and Eval：
- SemanticKITTI:

    `python infer.py -d /your_dataset -l /your_predictions_path -m trained_model -s valid/test`
    ```
    python infer.py -d /mnt/data/semantickitti -l HDC_result -m /mnt/data/dataset/'Final result'/512-594 -s valid
    python infer.py -d /mnt/data/semantickitti -l HDC_result -m pretrain0_4/senet-512 -s valid
    ```
    
    Eval for valid sequences:

    `python evaluate_iou.py -d /your_dataset -p /your_predictions_path`

    For test  sequences, need to upload to [CodaLab](https://competitions.codalab.org/competitions/20331#participate) pages.

- SemanticPOSS:

    `python infer_poss.py -d /your_dataset -l /your_predictions_path -m trained_model`

    This will generate both predictions and mIoU results.

<!-- ### Visualize Example:


- Visualize GT:

  `python visualize.py -w kitti/poss -d /your_dataset -s what_sequences`

- Visualize Predictions:

  `python visualize.py -w kitti/poss -d /your_dataset -p /your_predictions -s what_sequences` -->


## Pretrained Models and Logs:
| **KITTI Result** | **POSS Result** | **Ablation Study** | **Backbone HarDNet** |
| ---------------- | --------------- | ------------------ | -------------------- |
| [Google Drive](https://drive.google.com/file/d/167ofUNdkVnRoqZ28NAXRjykdVWnublUk/view?usp=share_link) | [Google Drive](https://drive.google.com/file/d/1DC66ky6k2aBpVg1Md1AR2tjqHzSYL5xC/view?usp=sharing) | [Google Drive](https://drive.google.com/file/d/1axrBYJflKMn0FLoC6HoN1G4RUmitIP1U/view?usp=sharing) | [Google Drive](https://drive.google.com/file/d/1afer0OX0WzxoMIWXV-btGVC7llt-4nUB/view?usp=sharing) |


## TODO List:
- [x] Release Pretrained Model and Logs.
- [ ] Try TensorRT acceleration.
- [ ] To make NLA adaptation framework, See [here](https://github.com/huixiancheng/SENet/blob/57d3e07777099c805fa27ceda68e359b2b7ae12d/modules/user.py#L178-L194).

## Acknowledgments：
Code framework derived from [SalsaNext](https://github.com/Halmstad-University/SalsaNext). Models are heavily based on [FIDNet](https://github.com/placeforyiming/IROS21-FIDNet-SemanticKITTI). Part of code from [SqueezeSegV3](https://github.com/chenfengxu714/SqueezeSegV3). Thanks to their open source code, and also to [Ph.D. Zhao](https://github.com/placeforyiming) for some helpful discussions.

## Citation：
~~~
@inproceedings{cheng2022cenet,
  title={Cenet: Toward Concise and Efficient Lidar Semantic Segmentation for Autonomous Driving},
  author={Cheng, Hui--Xian and Han, Xian--Feng and Xiao, Guo--Qiang},
  booktitle={2022 IEEE International Conference on Multimedia and Expo (ICME)},
  pages={01--06},
  year={2022},
  organization={IEEE}
}
~~~
