#!/bin/bash

task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
action_dim=${5}
gpu_id=${6}

head_camera_type=D435

# 训练参数 (Cross Attention需要调整显存)
BATCH_SIZE=128                          # 减小batch size (Cross Attention显存需求更大)
GRAD_ACCUM=2                            # 梯度累积，保持有效批次大小

DEBUG=False
save_ckpt=True

alg_name=robot_dp_${action_dim}_depth_crossattn
config_name=${alg_name}
addition_info=train
exp_name=${task_name}-robot_dp_depth_crossattn-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"

echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

if [ "$DEBUG" == "True" ]; then
    export WANDB_MODE=offline
else
    export WANDB_MODE=online
fi


export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}

if [ ! -d "./data/${task_name}-${task_config}-${expert_data_num}.zarr" ]; then
    bash process_data.sh ${task_name} ${task_config} ${expert_data_num}
fi

python train.py --config-name=${config_name}.yaml \
                            task.name=${task_name} \
                            task.dataset.zarr_path="data/${task_name}-${task_config}-${expert_data_num}.zarr" \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            dataloader.batch_size=${BATCH_SIZE} \
                            +dataloader.drop_last=True \
                            training.gradient_accumulate_every=${GRAD_ACCUM} \
                            exp_name=${exp_name} \
                            setting=${task_config} \
                            expert_data_num=${expert_data_num} \
                            head_camera_type=$head_camera_type \
                            hydra.run.dir="data/outputs/${task_name}_${config_name}_${seed}_$(date +%Y%m%d_%H%M%S)"
                            # logging.mode=${wandb_mode} \  
                            #checkpoint.save_ckpt=${save_ckpt}
