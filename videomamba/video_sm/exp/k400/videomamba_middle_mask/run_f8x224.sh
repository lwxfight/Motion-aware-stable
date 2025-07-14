export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

JOB_NAME='videomamba_middle_mask_ft_f8_res224'
OUTPUT_DIR="$(dirname $0)/$JOB_NAME"
LOG_DIR="./logs/${JOB_NAME}"
# PREFIX='../../tiny-kinetics-400/data/tiny-Kinetics-400'
# DATA_PATH='../../tiny-kinetics-400/annotations'
# DATA_PATH='/data/datasets/Drone/list'
DATA_PATH='../../list'
PREFIX='/data/datasets'


NUM_GPUS=1
# export CUDA_VISIBLE_DEVICES="0,1"

# torchrun --nproc-per-node=$NUM_GPUS\

python run_class_finetuning.py\
        --model videomamba_middle \
        --finetune ./pretrained_model/videomamba_m16_k400_mask_ft_f8_res224.pth \
        --data_path ${DATA_PATH} \
        --prefix ${PREFIX} \
        --data_set 'Kinetics_sparse' \
        --split ' ' \
        --nb_classes 13 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 1 \
        --num_sample 2 \
        --input_size 224 \
        --short_side_size 224 \
        --save_ckpt_freq 100 \
        --num_frames 8 \
        --num_workers 12 \
        --warmup_epochs 5 \
        --tubelet_size 1 \
        --epochs 91 \
        --lr 1e-3 \
        --layer_decay 0.8 \
        --drop_path 0.4 \
        --opt adamw \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.05 \
        --test_num_segment 4 \
        --test_num_crop 3 \
        --dist_eval \
        --test_best \
        --bf16 \
        --eval \
