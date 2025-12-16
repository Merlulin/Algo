import numpy as np
import gymnasium
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box, Dict

from ccp.planning import ResettablePlanner, PlannerConfig


class PreprocessorConfig(PlannerConfig):
    '''
    预处理配置类，继承自PlannerConfig，定义了预处理相关的参数。
    
    参数说明：
    - network_input_radius: 网络输入的观测半径（网格单元数），决定神经网络接收的观测范围
      默认5表示11x11的观测窗口（2*5+1）
    - intrinsic_target_reward: 内在奖励值，当智能体到达子目标（subgoal）时给予的奖励
      用于鼓励智能体跟随路径规划器提供的路径点
    
    继承的参数（来自PlannerConfig）：
    - use_static_cost: 是否使用静态代价（障碍物惩罚）
    - use_dynamic_cost: 是否使用动态代价（其他智能体位置惩罚）
    - reset_dynamic_cost: 是否在重置时重置动态代价
    '''
    network_input_radius: int = 5  # 网络输入的观测半径，默认5（11x11观测窗口）
    intrinsic_target_reward: float = 0.01  # 到达子目标时的内在奖励值


def ccp_preprocessor(env, algo_config):
    '''
    Follower算法的预处理函数，是预处理的主要入口点。
    
    作用：
    - 从algo_config中提取预处理配置
    - 调用wrap_preprocessors应用所有预处理包装器
    - 返回预处理后的环境
    
    参数:
        env: 基础环境实例（已经过create_env_base处理）
        algo_config: 算法配置对象，包含training_config.preprocessing配置
    
    返回:
        预处理后的环境实例，已应用所有Follower特有的预处理包装器
    
    使用场景：
    - 在example.py中用于推理时的环境预处理
    - 在eval.py中注册算法时使用
    '''
    env = wrap_preprocessors(env, algo_config.training_config.preprocessing)
    return env


def wrap_preprocessors(env, config: PreprocessorConfig, auto_reset=False):
    '''
    应用所有预处理包装器的核心函数，按照固定顺序包装环境。
    
    包装器应用顺序（从内到外）：
    1. FollowerWrapper: 核心包装器，添加路径规划和内在奖励机制
    2. CutObservationWrapper: 裁剪观测到网络输入大小
    3. ConcatPositionalFeatures: 拼接多个位置特征为单一obs张量
    4. AutoResetWrapper: 可选，自动重置环境（训练时使用）
    
    重要说明：
    - 包装器顺序很重要：内层包装器先处理观测，外层包装器对处理后的观测进行操作
    - FollowerWrapper在最内层，因为它需要原始的观测进行路径规划
    - CutObservationWrapper和ConcatPositionalFeatures在外层，对规划后的观测进行格式转换
    
    参数:
        env: 基础环境实例
        config: 预处理配置对象
        auto_reset: 是否启用自动重置（训练时通常为True，推理时为False）
    
    返回:
        完全预处理后的环境实例，可以直接用于训练或推理
    
    使用场景：
    - 训练时：在register_env.py的create_env()中调用，auto_reset=True
    - 推理时：在follower_preprocessor中调用，auto_reset=False
    '''
    env = CCPWrapper(env=env, config=config)  # 添加路径规划和内在奖励
    env = CutObservationWrapper(env, target_observation_radius=config.network_input_radius)  # 裁剪观测大小
    env = ConcatPositionalFeatures(env)  # 拼接位置特征
    if auto_reset:
        env = AutoResetWrapper(env)  # 自动重置（训练时使用）
    return env


class CCPWrapper(ObservationWrapper):
    '''
    Follower算法的核心包装器，实现了路径规划和内在奖励机制。
    
    核心功能：
    1. 路径规划：使用ResettablePlanner为每个智能体计算到目标的最短路径
    2. 子目标设置：将路径上的下一个点作为子目标，指导智能体移动
    3. 路径可视化：在观测中将规划路径标记为+1.0，障碍物标记为-1.0
    4. 内在奖励：当智能体到达子目标时给予奖励，鼓励跟随路径
    
    工作原理（"Learn to Follow"算法的关键）：
    - 使用A*等路径规划算法计算全局最优路径
    - 将长距离路径分解为短距离的子目标序列
    - 通过内在奖励引导智能体学习跟随路径规划器
    - 最终训练出的策略能够在无需规划器的情况下做出类似决策
    
    观测修改：
    - obstacles数组中：障碍物值从1变为-1，路径点设置为+1
    - 这样神经网络可以区分障碍物、路径和自由空间
    
    奖励修改：
    - 将环境奖励替换为内在奖励（到达子目标时给予intrinsic_target_reward）
    - 这实现了"Learning to Follow"的核心思想：模仿路径规划器的行为
    '''

    def __init__(self, env, config: PreprocessorConfig):
        super().__init__(env)
        self._cfg: PreprocessorConfig = config
        self.re_plan = ResettablePlanner(self._cfg)  # 可重置的路径规划器
        self.prev_goals = None  # 上一帧的子目标列表
        self.intrinsic_reward = None  # 内在奖励列表

    @staticmethod
    def get_relative_xy(x, y, tx, ty, obs_radius):
        '''
        将全局坐标转换为相对于智能体的局部观测坐标。
        
        坐标转换规则：
        - 智能体位于观测窗口的中心 (obs_radius, obs_radius)
        - 如果目标点超出观测范围，返回None
        - 转换公式：局部x = obs_radius - (全局x - 智能体x)
        
        参数:
            x, y: 智能体的全局坐标
            tx, ty: 目标点的全局坐标
            obs_radius: 观测半径
        
        返回:
            (局部x, 局部y) 或 (None, None) 如果目标超出范围
        '''
        dx, dy = x - tx, y - ty
        if dx > obs_radius or dx < -obs_radius or dy > obs_radius or dy < -obs_radius:
            return None, None
        return obs_radius - dx, obs_radius - dy

    def observation(self, observations):
        '''
        核心观测处理方法，为每个智能体进行路径规划和观测修改。
        
        工作流程：
        1. 更新路径规划器：基于当前观测更新动态代价（其他智能体位置）
        2. 获取路径：为每个智能体计算到目标的最短路径
        3. 子目标选择：从路径中选择下一个子目标（path[1]）
        4. 内在奖励计算：如果智能体到达了上一帧的子目标，给予奖励
        5. 观测修改：
           - 障碍物值：从1变为-1（便于区分）
           - 路径标记：将路径点标记为+1.0（在观测范围内）
        
        路径处理逻辑：
        - 如果路径为None（无可行路径）：使用最终目标作为子目标
        - 如果路径存在：使用path[1]作为子目标（path[0]是当前位置）
        - 只标记在观测范围内的路径点
        
        参数:
            observations: 原始观测列表，每个元素是智能体的观测字典
        
        返回:
            修改后的观测列表，包含路径标记和障碍物标记
        '''
        # 更新代价惩罚（基于当前观测，独立为每个智能体更新）
        self.re_plan.update(observations)

        # 获取每个智能体到全局目标的最短路径
        paths = self.re_plan.get_path()

        new_goals = []  # 存储每个智能体的新子目标
        intrinsic_rewards = []  # 存储每个智能体的内在奖励

        # 遍历智能体及其对应路径
        for k, path in enumerate(paths):
            obs = observations[k]

            # 检查是否有有效路径
            if path is None:
                new_goals.append(obs['target_xy'])  # 使用目标位置作为子目标
                path = []
            else:
                # 检查智能体是否到达了上一帧的子目标
                subgoal_achieved = self.prev_goals and obs['xy'] == self.prev_goals[k]
                # 如果到达子目标，给予内在奖励，否则为0
                intrinsic_rewards.append(self._cfg.intrinsic_target_reward if subgoal_achieved else 0.0)
                # 选择路径上的下一个点作为新的子目标
                new_goals.append(path[1])

            # 将观测中的障碍物值设置为-1.0（便于与路径区分）
            obs['obstacles'][obs['obstacles'] > 0] *= -1

            # 将路径添加到观测中，路径点设置为+1.0
            r = obs['obstacles'].shape[0] // 2  # 观测半径
            for idx, (gx, gy) in enumerate(path):
                x, y = self.get_relative_xy(*obs['xy'], gx, gy, r)  # 转换为局部坐标
                if x is not None and y is not None:  # 如果路径点在观测范围内
                    obs['obstacles'][x, y] = 1.0  # 标记为路径点
                else:
                    break  # 超出观测范围，停止标记
        # 更新上一帧的子目标和内在奖励，供下一帧使用
        self.prev_goals = new_goals
        self.intrinsic_reward = intrinsic_rewards

        return observations

    def get_intrinsic_rewards(self, reward):
        '''
        用内在奖励替换环境原始奖励。
        
        这是"Learn to Follow"算法的关键：通过内在奖励引导智能体学习跟随路径规划器。
        环境奖励被完全替换为基于路径跟随的内在奖励。
        
        参数:
            reward: 原始环境奖励列表
        
        返回:
            替换后的奖励列表（内在奖励）
        '''
        for agent_idx, r in enumerate(reward):
            reward[agent_idx] = self.intrinsic_reward[agent_idx]
        return reward

    def step(self, action):
        '''
        执行一步环境交互，应用观测处理和奖励替换。
        
        参数:
            action: 智能体动作列表
        
        返回:
            处理后的观测、内在奖励、完成标志、截断标志和信息
        '''
        observation, reward, done, tr, info = self.env.step(action)
        return self.observation(observation), self.get_intrinsic_rewards(reward), done, tr, info

    def reset_state(self):
        '''
        重置包装器状态，初始化路径规划器和历史记录。
        
        工作内容：
        1. 重置路径规划器状态
        2. 向规划器添加全局障碍物和智能体初始位置
        3. 清空上一帧的子目标和奖励记录
        '''
        self.re_plan.reset_states()  # 重置规划器
        # 添加全局障碍物和智能体初始位置到规划器
        self.re_plan._agent.add_grid_obstacles(self.get_global_obstacles(), self.get_global_agents_xy())

        self.prev_goals = None  # 清空上一帧的子目标
        self.intrinsic_reward = None  # 清空奖励记录

    def reset(self, **kwargs):
        '''
        重置环境并初始化包装器状态。
        
        参数:
            **kwargs: 重置参数
        
        返回:
            处理后的观测和信息字典
        '''
        observations, infos = self.env.reset(**kwargs)
        self.reset_state()  # 重置包装器状态
        return self.observation(observations), infos  # 返回处理后的观测


class CutObservationWrapper(ObservationWrapper):
    '''
    观测裁剪包装器，将观测从原始大小裁剪到网络输入大小。
    
    作用：
    - 环境可能提供较大的观测范围（如obs_radius=5，11x11）
    - 神经网络只需要较小的输入（如network_input_radius=5，也是11x11，或更小）
    - 从观测中心裁剪出指定大小的区域
    
    工作原理：
    - 计算观测中心位置
    - 从中心向四周裁剪target_observation_radius大小的区域
    - 只裁剪形状为(d, d)的观测（如obstacles、agents等位置特征）
    - 其他类型的观测保持不变
    
    使用场景：
    - 当环境的观测半径大于网络输入半径时使用
    - 减少网络输入大小，降低计算量和内存占用
    - 保留以智能体为中心的关键信息
    '''
    def __init__(self, env, target_observation_radius):
        '''
        初始化裁剪包装器，更新观测空间定义。
        
        参数:
            target_observation_radius: 目标观测半径（网络输入半径）
        '''
        super().__init__(env)
        self._target_obs_radius = target_observation_radius  # 目标观测半径
        self._initial_obs_radius = self.env.observation_space['obstacles'].shape[0] // 2  # 原始观测半径

        # 更新观测空间：将所有(d, d)形状的观测裁剪到目标大小
        for key, value in self.observation_space.items():
            d = self._initial_obs_radius * 2 + 1  # 原始观测大小
            if value.shape == (d, d):  # 如果是位置特征（二维数组）
                r = self._target_obs_radius
                self.observation_space[key] = Box(0.0, 1.0, shape=(r * 2 + 1, r * 2 + 1))  # 更新为裁剪后的大小

    def observation(self, observations):
        '''
        裁剪观测到目标大小，从中心区域提取。
        
        裁剪公式：
        - 中心位置：ir (初始观测半径)
        - 裁剪范围：[ir - tr : ir + tr + 1]，即从中心向四周裁剪tr半径的区域
        - 只裁剪形状为(d, d)的数组（位置特征）
        
        参数:
            observations: 原始观测列表
        
        返回:
            裁剪后的观测列表
        '''
        tr = self._target_obs_radius  # 目标半径
        ir = self._initial_obs_radius  # 初始半径
        d = ir * 2 + 1  # 初始观测大小

        for obs in observations:
            for key, value in obs.items():
                if hasattr(value, 'shape') and value.shape == (d, d):  # 如果是位置特征
                    # 从中心裁剪：中心位置为ir，裁剪范围[ir-tr, ir+tr+1]
                    obs[key] = value[ir - tr:ir + tr + 1, ir - tr:ir + tr + 1]

        return observations


class ConcatPositionalFeatures(ObservationWrapper):
    '''
    位置特征拼接包装器，将多个位置特征（如obstacles、agents）拼接为单一obs张量。
    
    作用：
    - 将多个二维位置特征（obstacles, agents, agents_global等）拼接为一个三维张量
    - 便于神经网络处理：输入格式为(channels, height, width)
    - 保持非位置特征（如xy坐标）在观测字典中，不被拼接
    
    拼接顺序（通过key_comparator排序）：
    1. obstacles（优先级0）：障碍物地图，最重要
    2. agents相关（优先级1）：智能体位置信息
    3. 其他位置特征（优先级2）：其他二维特征
    
    输出格式：
    - obs: 形状为(n_features, height, width)的三维数组，包含所有位置特征
    - 其他键值对（如xy, target_xy）保留在观测字典中，作为额外信息
    '''

    def __init__(self, env):
        '''
        初始化拼接包装器，识别需要拼接的位置特征并更新观测空间。
        
        工作流程：
        1. 识别所有形状为(full_size, full_size)的特征（位置特征）
        2. 将这些特征标记为需要拼接
        3. 创建新的观测空间，包含拼接后的obs张量
        4. 按照key_comparator排序，确保拼接顺序一致
        '''
        super().__init__(env)
        self.to_concat = []  # 需要拼接的键列表

        observation_space = Dict()
        full_size = self.env.observation_space['obstacles'].shape[0]  # 观测大小

        # 识别需要拼接的位置特征
        for key, value in self.observation_space.items():
            if value.shape == (full_size, full_size):  # 如果是位置特征（二维数组）
                self.to_concat.append(key)
            else:
                observation_space[key] = value  # 非位置特征保留在字典中

        # 创建拼接后的obs张量：形状为(n_features, height, width)
        obs_shape = (len(self.to_concat), full_size, full_size)
        observation_space['obs'] = Box(0.0, 1.0, shape=obs_shape)
        self.to_concat.sort(key=self.key_comparator)  # 按优先级排序
        self.observation_space = observation_space

    def observation(self, observations):
        '''
        拼接位置特征为obs张量，并清理原始特征。
        
        工作流程：
        1. 对每个智能体的观测，按顺序拼接所有位置特征
        2. 从原始观测字典中删除已拼接的特征
        3. 将其他特征转换为float32类型
        4. 将拼接后的obs张量添加到观测字典
        
        参数:
            observations: 原始观测列表
        
        返回:
            拼接后的观测列表，包含obs张量和其他特征
        '''
        for agent_idx, obs in enumerate(observations):
            # 按顺序拼接所有位置特征：[None]增加一个维度用于拼接
            main_obs = np.concatenate([obs[key][None] for key in self.to_concat], axis=0)
            # 删除已拼接的原始特征
            for key in self.to_concat:
                del obs[key]

            # 将剩余特征转换为float32
            for key in obs:
                obs[key] = np.array(obs[key], dtype=np.float32)
            # 添加拼接后的obs张量
            observations[agent_idx]['obs'] = main_obs.astype(np.float32)
        return observations

    @staticmethod
    def key_comparator(x):
        '''
        键比较器，用于确定位置特征的拼接顺序。
        
        排序规则：
        - obstacles: 优先级0（最重要，第一个通道）
        - 包含'agents'的键: 优先级1（智能体相关特征）
        - 其他: 优先级2（其他位置特征）
        
        参数:
            x: 特征键名
        
        返回:
            排序键字符串，用于确定拼接顺序
        '''
        if x == 'obstacles':
            return '0_' + x  # obstacles优先级最高
        elif 'agents' in x:
            return '1_' + x  # agents相关次之
        return '2_' + x  # 其他特征最后


class AutoResetWrapper(gymnasium.Wrapper):
    '''
    自动重置包装器，当所有智能体完成时自动重置环境。
    
    作用：
    - 训练时自动处理回合结束，无需手动重置
    - 确保训练循环的连续性
    - 在回合结束时立即返回新回合的初始观测
    
    工作原理：
    - 检测所有智能体是否完成（all(terminated) 或 all(truncated)）
    - 如果完成，自动调用reset()获取新回合的初始观测
    - 返回新的观测，但保持原来的奖励和完成标志
    
    注意事项：
    - 只在训练时使用（wrap_preprocessors中auto_reset=True）
    - 推理时通常不使用，需要手动控制重置时机
    - 重置后返回的observations是新的，但rewards和flags仍是上一回合的
    
    使用场景：
    - Sample Factory训练时自动重置，提高训练效率
    - 批量环境训练时，自动处理各个环境的完成状态
    '''
    def step(self, action):
        '''
        执行一步环境交互，如果回合结束则自动重置。
        
        参数:
            action: 智能体动作列表
        
        返回:
            - observations: 如果回合结束则为新回合的初始观测，否则为当前观测
            - rewards, terminated, truncated, infos: 当前步骤的返回值
        '''
        observations, rewards, terminated, truncated, infos = self.env.step(action)
        # 如果所有智能体都完成（terminated或truncated），自动重置环境
        if all(terminated) or all(truncated):
            observations, _ = self.env.reset()  # 获取新回合的初始观测
        return observations, rewards, terminated, truncated, infos
