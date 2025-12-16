import time

import numpy as np
from pogema import AnimationConfig, AnimationMonitor

from pogema import pogema_v0

from follower.training_config import Environment

import gymnasium
import re
from copy import deepcopy
from pogema import GridConfig

from env.custom_maps import MAPS_REGISTRY
from follower.preprocessing import wrap_preprocessors, PreprocessorConfig


class ProvideGlobalObstacles(gymnasium.Wrapper):
    '''
    全局信息提供包装器，为环境添加获取全局障碍物和智能体位置的方法。
    
    作用：
    - 在部分可观测（POMAPF）环境中，智能体通常只能看到局部观测
    - 某些算法（如C++版本的Follower）需要全局地图信息来进行路径规划
    - 这个包装器提供了访问全局信息的接口，而不改变环境的基本行为
    
    提供的方法：
    - get_global_obstacles(): 获取整个地图的障碍物布局（二维列表）
    - get_global_agents_xy(): 获取所有智能体的当前位置坐标
    
    使用场景：
    - 在follower_cpp/preprocessing.py中的ProvideMapWrapper会调用这些方法
    - 将全局信息添加到观测中，供C++推理算法使用
    '''
    def get_global_obstacles(self):
        '''
        获取全局障碍物地图。
        
        返回:
            二维列表，表示整个地图的障碍物布局（1表示障碍物，0表示可通过）
        '''
        return self.grid.get_obstacles().astype(int).tolist()

    def get_global_agents_xy(self):
        '''
        获取所有智能体的全局坐标位置。
        
        返回:
            智能体坐标列表，每个元素是(x, y)坐标元组
        '''
        return self.grid.get_agents_xy()


def create_env_base(config: Environment):
    '''
    创建基础环境实例，应用必要的包装器。
    
    工作流程（按顺序应用包装器）：
    1. 创建Pogema基础环境（pogema_v0）
    2. 添加ProvideGlobalObstacles包装器：提供全局信息访问接口
    3. 如果use_maps=True：添加MultiMapWrapper，支持多地图课程学习训练
    4. 如果with_animation=True：添加AnimationMonitor，保存动画到renders目录
    5. 添加RuntimeMetricWrapper：记录环境运行时间指标
    
    包装器应用顺序很重要：
    - 内层包装器会先处理环境交互
    - 外层包装器可以访问内层包装器的功能
    - 例如：RuntimeMetricWrapper在最外层，可以测量整个环境（包括动画）的运行时间
    
    参数:
        config: Environment配置对象，包含环境的所有配置参数（地图、智能体、动画等）
    
    返回:
        完整配置的环境实例，已应用所有必要的包装器
    
    使用场景：
    - 在follower/register_env.py中被create_env()调用
    - 在example.py中直接调用创建推理环境
    '''
    env = pogema_v0(grid_config=config.grid_config)  # 创建pogema基础环境
    env = ProvideGlobalObstacles(env)  # 添加全局信息访问接口
    if config.use_maps:
        env = MultiMapWrapper(env)  # 支持多地图训练（根据map_name模式匹配）
    if config.with_animation:
        env = AnimationMonitor(env, AnimationConfig(directory='renders', egocentric_idx=None))  # 保存动画到renders目录下

    # adding runtime metrics
    env = RuntimeMetricWrapper(env)  # 记录运行时间指标

    return env


class RuntimeMetricWrapper(gymnasium.Wrapper):
    '''
    运行时指标包装器，记录环境的运行时间统计信息。
    
    作用：
    - 测量每个回合的总运行时间（不包括环境step的时间）
    - 将运行时间作为指标添加到infos字典中，用于性能分析和日志记录
    - 帮助识别环境性能瓶颈和优化机会
    
    工作原理：
    - reset时：记录回合开始时间，重置step时间累计器
    - step时：累计每次step的执行时间（环境计算时间，不包括重置时间）
    - 回合结束时：计算总运行时间（当前时间 - 开始时间 - 累计step时间），添加到metrics中
    
    时间统计说明：
    - runtime = 总时间 - 环境step累计时间
    - 这个runtime主要反映环境的初始化、重置等开销，不包括step计算时间
    
    使用场景：
    - 训练和评估时自动记录，用于性能监控
    - 在wandb或日志中可以看到每个回合的运行时间指标
    '''
    def __init__(self, env):
        super().__init__(env)
        self._start_time = None  # 回合开始时间
        self._env_step_time = None  # 累计的环境step执行时间

    def step(self, actions):
        '''
        执行一步环境交互，记录step执行时间。
        
        当回合结束时（所有智能体terminated或truncated），计算总运行时间
        并将其添加到infos[0]['metrics']['runtime']中。
        
        参数:
            actions: 智能体动作列表
        
        返回:
            标准的gymnasium step返回值，infos中包含runtime指标（如果回合结束）
        '''
        env_step_start = time.monotonic()
        observations, rewards, terminated, truncated, infos = self.env.step(actions)
        env_step_end = time.monotonic()
        self._env_step_time += env_step_end - env_step_start  # 累计step执行时间
        if all(terminated) or all(truncated):
            # 回合结束，计算总运行时间（不包括step时间）
            final_time = time.monotonic() - self._start_time - self._env_step_time
            if 'metrics' not in infos[0]:
                infos[0]['metrics'] = {}
            infos[0]['metrics'].update(runtime=final_time)  # 添加到指标字典
        return observations, rewards, terminated, truncated, infos

    def reset(self, **kwargs):
        '''
        重置环境，初始化时间统计。
        
        参数:
            **kwargs: 重置参数
        
        返回:
            重置后的观测
        '''
        obs = self.env.reset(**kwargs)
        self._start_time = time.monotonic()  # 记录回合开始时间
        self._env_step_time = 0.0  # 重置step时间累计器
        return obs


class MultiMapWrapper(gymnasium.Wrapper):
    '''
    多地图包装器，支持在多张地图上进行训练和评估。
    
    主要用途：
    - 数据增强：通过在多张不同的地图上训练，提高模型的泛化能力
    - 课程学习：可以逐步增加地图难度
    - 随机化训练：每次reset时随机选择地图，增加训练的多样性
    
    工作原理：
    - 初始化时：根据grid_config.map_name的正则表达式模式，从MAPS_REGISTRY中匹配所有符合条件的地图
    - reset时：从匹配的地图列表中随机选择一张地图，更新环境配置并重置环境
    - 使用独立的随机数生成器，确保地图选择的随机性不受其他随机操作影响
    
    地图匹配机制：
    - 使用正则表达式匹配：grid_config.map_name可以是正则表达式模式
    - 例如：map_name='mazes-.+' 会匹配所有以'mazes-'开头的迷宫地图
    - 例如：map_name='(mazes-s[0-9]_|mazes-s[1-3][0-9]_)' 匹配mazes-s0到mazes-s39的地图
    
    使用场景：
    - 训练时：enable use_maps=True，在多个地图上随机训练
    - 评估时：可以指定特定的地图列表进行评估
    
    注意事项：
    - 如果没有匹配到任何地图，会抛出KeyError异常
    - 每次reset都会随机选择地图，确保训练过程中的多样性
    '''
    def __init__(self, env):
        '''
        初始化多地图包装器，根据map_name模式匹配所有符合条件的地图。
        
        工作流程：
        1. 从环境配置中获取map_name正则表达式模式
        2. 遍历MAPS_REGISTRY中的所有地图，使用re.match匹配
        3. 为每个匹配的地图创建GridConfig配置对象
        4. 将所有匹配的地图配置保存在self._configs列表中
        '''
        super().__init__(env)
        self._configs = []  # 存储所有匹配的地图配置
        self._rnd = np.random.default_rng(self.grid_config.seed)  # 随机数生成器
        pattern = self.grid_config.map_name  # 获取地图名称的正则表达式模式

        if pattern:
            # 遍历地图注册表，匹配符合pattern的地图
            for map_name in sorted(MAPS_REGISTRY):
                if re.match(pattern, map_name):
                    cfg = deepcopy(self.grid_config)
                    cfg.map = MAPS_REGISTRY[map_name]  # 设置具体的地图数据
                    cfg.map_name = map_name  # 设置具体的地图名称
                    cfg = GridConfig(**cfg.dict())  # 转换为GridConfig对象
                    self._configs.append(cfg)
            if not self._configs:
                raise KeyError(f"No map matching: {pattern}")  # 没有匹配到任何地图

    def reset(self, seed=None, **kwargs):
        '''
        重置环境，随机选择一张地图并更新环境配置。
        
        工作流程：
        1. 使用新的seed初始化随机数生成器
        2. 从匹配的地图列表中随机选择一张地图
        3. 更新环境的grid_config为选中的地图配置
        4. 调用底层环境的reset方法重置环境
        
        参数:
            seed: 随机种子，用于确保可复现性
            **kwargs: 其他重置参数
        
        返回:
            重置后的观测信息
        
        注意：
        - 每次reset都会选择不同的地图（随机），增加训练多样性
        - seed参数会影响地图选择的随机性，相同seed会产生相同的选择序列
        '''
        self._rnd = np.random.default_rng(seed)  # 使用新的seed初始化随机数生成器
        if self._configs is not None and len(self._configs) >= 1:
            map_idx = self._rnd.integers(0, len(self._configs))  # 随机选择地图索引
            cfg = deepcopy(self._configs[map_idx])  # 复制选中的地图配置
            self.env.unwrapped.grid_config = cfg  # 更新环境的网格配置
            self.env.unwrapped.grid_config.seed = seed  # 设置地图的随机种子
        return self.env.reset(seed=seed, **kwargs)  # 重置环境


def main():
    '''
    测试函数，用于验证环境创建和基本功能。
    
    功能：
    - 创建默认配置的环境实例
    - 应用预处理包装器
    - 重置环境
    - 渲染环境（如果支持）
    
    使用方式：
    - 直接运行此文件：python env/create_env.py
    - 用于快速测试环境是否正常工作
    - 调试环境配置和包装器是否正确应用
    '''
    env = create_env_base(config=Environment())  # 创建基础环境
    env = wrap_preprocessors(env, config=PreprocessorConfig())  # 应用预处理包装器
    env.reset()  # 重置环境
    env.render()  # 渲染环境（如果支持）


if __name__ == '__main__':
    main()
