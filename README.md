<div align="center">
<h2>
Motion-Consistent Representation Learning for
UAV-Based Action Recognition
</h2>
</div>

## Usage
### 1. Install
- Clone this repo:
  ```shell
  git clone  https://github.com/lwxfight/-StaRNet
  cd StaRNet 
  ```

- Create Conda environments
  ```shell
  conda create -n  starnet python=3.10
  conda activate starnet
  ```


- Install PyTorch 2.1.1+cu118
  ```shell
  pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
  ```

- Install `causal_conv1d` and `mamba`
  ```shell
  pip install -r requirements.txt
  pip install -e causal-conv1d
  pip install -e mamba
  ```
We use the pretrained weight to initialize the model. You can download it from  [VideoMamba MODEL ZOO](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_m16_k400_f16_res224.pth) and place it in `./videomamba/video_sm/pretrained_model/`.

### 2. Prepare Drone datasets
Please follow the instruction of [Drone Action](https://asankagp.github.io/droneaction/) and [UAV Human](https://drive.google.com/drive/folders/1oEnsidJMOjeK33zHT4EU32W2D4fIeilf) to prepare the dataset. Then manully summarize all the paths and labels in the following format in `train.csv`, `test.csv` and `val.csv` under `list` folder.
```
file_name1 label1
file_name2 label2
file_name3 label3
...
```

### 3. Train

For training, you can simply run the script as follows:
```shell
cd ./videomamba/video_sm
bash ./exp/k400/videomamba_middle_mask/run_f8x224.sh
```

> **Notes:**
> 1. Chage `DATA_PATH`, `PREFIX` and `nb_classes` to your data path before running the scripts.
> 2. Set `--finetune` when using masked pretrained model.

### 4. Eval
Run the following script to eval:
```bash
bash ./exp/k400/videomamba_middle_mask/eval.sh
```


