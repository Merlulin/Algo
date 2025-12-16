import numpy as np
from sample_factory.utils.typing import Env
from sample_factory.envs.env_utils import register_env

from env.create_env import create_env_base
from follower.training_config import Experiment

import gymnasium
from follower.training_config import Environment
from follower.preprocessing import PreprocessorConfig, wrap_preprocessors


def create_env(environment_cfg: Environment, preprocessing_cfg: PreprocessorConfig):
    '''
    创建单个环境实例，并应用预处理包装器。
    
    工作流程：
    1. 使用environment_cfg创建基础环境（包含地图、智能体配置等）
    2. 应用预处理包装器（添加观测预处理、奖励处理等）
    3. 返回完整配置的环境实例
    
    参数:
        environment_cfg: 环境配置对象，包含地图、智能体数量、观测半径等配置
        preprocessing_cfg: 预处理配置对象，定义观测预处理方式
    
    返回:
        配置好的环境实例，可以直接用于训练或推理
    '''
    env = create_env_base(environment_cfg)
    env = wrap_preprocessors(env, config=preprocessing_cfg, auto_reset=True)  # 应用预处理包装器（添加观测预处理、奖励处理等）也就是论文的关键预处理策略（规划+观测拼接+裁剪）
    return env


class MultiEnv(gymnasium.Wrapper):
    '''
    多环境包装器类，将多个环境实例组合成一个统一的环境接口。
    
    主要用途：
    - 支持大规模智能体训练：当target_num_agents > grid_config.num_agents时，
      通过创建多个子环境来实现大规模智能体训练（课程学习场景）
    - 统一接口：将多个环境的行为（step、reset等）合并为单个环境的接口，
      对上层代码透明
    
    工作原理：
    - 如果target_num_agents为None，只创建单个环境
    - 如果target_num_agents不为None，创建多个子环境，每个子环境的智能体数量
      由grid_config.num_agents决定，子环境数量 = target_num_agents // num_agents
    - step和reset操作会遍历所有子环境，合并它们的返回结果
    
    使用场景：
    - 课程学习：不同env_id可以使用不同的agent_bins值，实现从少到多的智能体数量训练
    - 大规模训练：通过多个子环境并行处理，支持更多智能体的训练
    '''
    def __init__(self, env_cfg: Environment, preprocessing_cfg: PreprocessorConfig):
        if env_cfg.target_num_agents is None:
            self.envs = [create_env(env_cfg, preprocessing_cfg)]
        else:
            assert env_cfg.target_num_agents % env_cfg.grid_config.num_agents == 0, \
                "Target num follower must be divisible by num agents"
            num_envs = env_cfg.target_num_agents // env_cfg.grid_config.num_agents
            self.envs = [create_env(env_cfg, preprocessing_cfg) for _ in range(num_envs)]

        super().__init__(self.envs[0])

    def step(self, actions):
        '''
        执行一步环境交互，遍历所有子环境并合并结果。
        
        参数:
            actions: 所有智能体的动作列表，需要按照子环境的顺序和智能体数量分组
        
        返回:
            合并后的观测、奖励、完成标志、截断标志和信息字典
        '''
        obs, rewards, dones, truncated, infos = [], [], [], [], []
        last_agents = 0
        for env in self.envs:
            env_num_agents = env.get_num_agents()
            action = actions[last_agents: last_agents + env_num_agents]
            last_agents = last_agents + env_num_agents
            o, r, d, t, i = env.step(action)
            obs += o
            rewards += r
            dones += d
            truncated += t
            infos += i
        return obs, rewards, dones, truncated, infos

    def reset(self, seed, **kwargs):
        '''
        重置所有子环境，使用不同的种子确保环境间的多样性。
        
        参数:
            seed: 基础随机种子
            **kwargs: 其他重置参数
        
        返回:
            合并后的观测列表和空字典（符合gymnasium接口规范）
        '''
        obs = []
        for idx, env in enumerate(self.envs):
            inner_seed = seed + idx
            o, _ = env.reset(seed=inner_seed, **kwargs)
            obs += o
        return obs, {}

    def sample_actions(self):
        '''
        从所有子环境中采样动作，用于随机动作测试。
        
        返回:
            所有智能体的随机动作数组
        '''
        actions = []
        for env in self.envs:
            actions += list(env.sample_actions())
        return np.array(actions)

    @property
    def num_agents(self):
        '''
        获取所有子环境中的智能体总数。
        
        返回:
            所有子环境的智能体数量之和
        '''
        return sum([env.get_num_agents() for env in self.envs])

    def render(self):
        '''
        渲染所有子环境，用于可视化调试。
        '''
        for q in self.envs:
            q.render()


def make_env(full_env_name, cfg=None, env_config=None, render_mode=None):
    '''
    Sample Factory框架需要的环境工厂函数，根据配置创建环境实例。
    
    工作流程：
    1. 从cfg中解析Experiment配置对象
    2. 提取environment和preprocessing配置
    3. 如果配置了agent_bins和target_num_agents，启用课程学习模式：
       - 根据env_id从agent_bins中选择智能体数量（循环选择）
       - 创建MultiEnv包装器支持多环境组合
    4. 否则创建单个环境实例
    
    课程学习机制：
    - agent_bins定义了不同env_id对应的智能体数量列表
    - env_id % len(agent_bins) 确保env_id循环选择agent_bins中的值
    - 例如：agent_bins=[64, 128, 256], env_id=5 → 选择agent_bins[5%3=2]=256
    
    参数:
        full_env_name: 环境名称（Sample Factory框架传入）
        cfg: Sample Factory配置对象，包含所有训练参数
        env_config: 环境特定配置（worker_index, vector_index, env_id等）
        render_mode: 渲染模式（未使用）
    
    返回:
        - MultiEnv实例：当启用课程学习时（agent_bins和target_num_agents均不为None）
        - 单个环境实例：其他情况
    '''
    p_config = Experiment(**vars(cfg))
    environment_config = p_config.environment
    preprocessing_config = p_config.preprocessing
    # todo make this code simpler

    if environment_config.agent_bins is not None and environment_config.target_num_agents is not None:
        # 课程学习模式：根据env_id从agent_bins中选择智能体数量
        if environment_config.env_id is None:
            num_agents = environment_config.agent_bins[0]
        else:
            num_agents = environment_config.agent_bins[environment_config.env_id % len(environment_config.agent_bins)]
        environment_config.grid_config.num_agents = num_agents
        # 用MultiEnv把多个小环境拼成一个“大 batch”并行（把 actions/obs/reward 拼接起来）
        return MultiEnv(environment_config, preprocessing_config)
    # 创建单一的训练环境
    return create_env(environment_config, preprocessing_config)


class CustomEnv:
    '''
    Sample Factory框架需要的环境工厂类。
    
    作用：
    - 作为Sample Factory框架和环境创建函数之间的适配器
    - Sample Factory的register_env函数需要一个包含make_env方法的对象
    - 这个类封装了make_env函数，使其符合Sample Factory的接口要求
    
    使用方式：
    - Sample Factory框架会调用CustomEnv().make_env()来创建环境实例
    - 每个worker进程和vector环境都会调用此方法创建自己的环境实例
    '''
    def make_env(self, env_name, cfg, env_config, render_mode) -> Env:
        '''
        Sample Factory框架调用的环境创建方法。
        
        参数:
            env_name: 环境名称
            cfg: Sample Factory配置对象
            env_config: 环境配置（包含worker_index, vector_index, env_id等）
            render_mode: 渲染模式
        
        返回:
            创建的环境实例
        '''
        return make_env(env_name, cfg, env_config, render_mode)


def register_pogema_envs(env_name):
    '''
    注册Pogema环境到Sample Factory框架。
    
    工作内容：
    1. 创建CustomEnv工厂对象
    2. 调用Sample Factory的register_env函数，将环境名称和工厂方法注册到框架中
    3. 注册后，Sample Factory可以通过env_name找到对应的环境创建函数
    
    参数:
        env_name: 环境名称标识符（如'PogemaMazes-v0'）
    
    使用场景：
    - 在训练开始时调用，让Sample Factory框架知道如何创建指定的环境
    - 必须在创建runner之前调用，否则框架无法找到环境创建函数
    '''
    env_factory = CustomEnv()
    register_env(env_name, env_factory.make_env)


def register_custom_components(env_name):
    '''
    注册自定义环境组件到Sample Factory框架（统一入口函数）。
    '''
    register_pogema_envs(env_name)
