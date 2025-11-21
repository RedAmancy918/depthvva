import cv2
import numpy as np
from .dp_model import DP
import yaml


def encode_obs(observation):
    # === 读取 RGB ===
    head_cam = np.moveaxis(observation["observation"]["head_camera"]["rgb"], -1, 0) / 255.0
    left_cam = np.moveaxis(observation["observation"]["left_camera"]["rgb"], -1, 0) / 255.0
    right_cam = np.moveaxis(observation["observation"]["right_camera"]["rgb"], -1, 0) / 255.0

    # === 从 observation 中读取 graydepth（由 get_obs 生成）===
    depth_gray = observation["observation"]["head_camera"]["graydepth"]
    # 确保维度统一为 (3,H,W)
    if depth_gray.ndim == 2:
        depth_gray = np.expand_dims(depth_gray, -1)
    if depth_gray.shape[2] == 1:
        depth_gray = np.repeat(depth_gray, 3, axis=2)
    depth_gray = np.moveaxis(depth_gray, -1, 0)

    # === 拼装模型输入 ===
    obs = dict(
        head_cam=head_cam,
        head_camera_depth=depth_gray,
        agent_pos=observation["joint_action"]["vector"],
        # left_cam=left_cam,
        # right_cam=right_cam,
    )
    return obs



def get_model(usr_args):
    ckpt_file = f"./policy/DP_fusion/checkpoints/{usr_args['task_name']}-{usr_args['ckpt_setting']}-{usr_args['expert_data_num']}-{usr_args['seed']}/{usr_args['checkpoint_num']}.ckpt"
    action_dim = usr_args['left_arm_dim'] + usr_args['right_arm_dim'] + 2 # 2 gripper
    
    load_config_path = f'./policy/DP/diffusion_policy/config/robot_dp_{action_dim}.yaml'
    with open(load_config_path, "r", encoding="utf-8") as f:
        model_training_config = yaml.safe_load(f)
    
    n_obs_steps = model_training_config['n_obs_steps']
    n_action_steps = model_training_config['n_action_steps']

    return DP(ckpt_file, n_obs_steps=n_obs_steps, n_action_steps=n_action_steps)


def eval(TASK_ENV, model, observation):
    """
    TASK_ENV: Task Environment Class, you can use this class to interact with the environment
    model: The model from 'get_model()' function
    observation: The observation about the environment
    """
    obs = encode_obs(observation)
    instruction = TASK_ENV.get_instruction()

    actions = model.get_action(obs)

    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        model.update_obs(obs)

def reset_model(model):
    model.reset_obs()

