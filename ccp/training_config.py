from typing import Optional, Union

from ccp.model import EncoderConfig
from ccp.preprocessing import PreprocessorConfig

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from pogema import GridConfig
from pydantic import BaseModel # 数据验证工具，定义的数据类型基于BaseModel之后可以使用注解定义字段。


class DecMAPFConfig(GridConfig):
    '''
    继承于pogema的网格地图配置信息
    '''
    integration: Literal['SampleFactory'] = 'SampleFactory' # 指定训练使用Sample Factory框架
    collision_system: Literal['priority', 'block_both', 'soft'] = 'soft' # 碰撞处理的策略：优先级高有限，block_both同时阻止，soft软碰撞。
    observation_type: Literal['POMAPF'] = 'POMAPF' # 观测类型：POMAPF，就是局部观测，只能观测局部范围的信息。
    auto_reset: Literal[False] = False # 是否自动重置环境。

    num_agents: int = 64 # 环境中的智能体数量。
    obs_radius: int = 5 # 观测半径，智能体可以观测到的范围。
    max_episode_steps: int = 512 # 每个episode的最大步数， 也就是智能体在环境中可以移动的最大步数。
    map_name: str = '(wc3-[A-P]|sc1-[A-S]|sc1-TaleofTwoCities|street-[A-P]|mazes-s[0-9]_|mazes-s[1-3][0-9]_|random-s[0-9]_|random-s[1-3][0-9]_)' # 地图名称的正则表达式，用于匹配地图名称。


class Environment(BaseModel, ):
    '''基础环境的配置'''

    grid_config: DecMAPFConfig = DecMAPFConfig() # 地图环境，使用pogema的网格地图配置
    env: Literal['PogemaMazes-v0'] = "PogemaMazes-v0" # 注册具体选用的环境
    with_animation: bool = False # 是否开启动画显示。
    worker_index: int = None # Sample Factory 多进程训练，工作进程索引
    vector_index: int = None # 在向量化环境中标识环境实例
    env_id: int = None # 环境实例ID
    target_num_agents: Optional[int] = None # 目标智能体数量
    agent_bins: Optional[list] = [64, 128, 256, 256] # 智能体数量列表
    use_maps: bool = True # 是否使用预定义地图。 true 下 启用 MultiMapWrapper，从 MAPS_REGISTRY 中根据 map_name 模式匹配加载地图

    every_step_metrics: bool = False # 是否每个步骤都记录指标。 False 仅在回合结束时记录


class EnvironmentMazes(Environment):
    # 针对迷宫场景的特殊调配，只使用迷宫地图，使用更大的 agent_bins 起始值
    env: Literal['PogemaMazes-v0'] = "PogemaMazes-v0"
    use_maps: bool = True
    target_num_agents: Optional[int] = 256
    agent_bins: Optional[list] = [128, 256, 256, 256]
    grid_config: DecMAPFConfig = DecMAPFConfig(on_target='restart', max_episode_steps=512,
                                               map_name=r'mazes-.+') # 覆盖网格配置，修改为终身问题，原始GridConfig是使用的Mapf，on_target是finish


class Experiment(BaseModel):
    '''
    实验配置类，包含实验的各个配置项。
    
    重点配置参数说明：
    
    【并行与采样配置】
    - rollout (默认8): 每个环境实例在收集数据前执行的最大步数，控制经验回放缓冲区大小
    - num_workers (默认4): 并行采样进程数，实际训练常用8，更多workers提高采样速度
    - num_envs_per_worker (默认4): 每个worker并行运行的环境实例数，总并行环境数 = num_workers × num_envs_per_worker
    
    【模型架构配置】
    - use_rnn (默认False): 是否使用RNN，False使用MLP无记忆能力，True使用RNN/GRU处理时序依赖
    - rnn_size (默认256): RNN隐藏层维度，仅在use_rnn=True时有效
    - recurrence (默认8): RNN展开长度（训练时的序列长度），影响梯度传播距离和内存占用
    
    【训练超参数】
    - learning_rate (默认0.000146): 学习率，实际训练中Follower约0.00022，FollowerLite约0.00013
    - gamma (默认0.965): 折扣因子，范围[0,1]越大越重视长期奖励，实际训练中约0.97-0.98
    - batch_size (默认2048): 每次梯度更新的样本数量，实际训练常用16384
    - ppo_clip_ratio (默认0.1): PPO裁剪比例，限制策略更新幅度，实际训练常用0.2
    - exploration_loss_coeff (默认0.018): 探索损失系数，鼓励策略探索，实际训练约0.015-0.023
    - num_batches_per_epoch (默认16): 每个训练epoch的批次数量
    
    【优化器配置】
    - optimizer (默认'adam'): 优化器类型，'adam'为Adam优化器，'lamb'为LAMB优化器适合大batch训练
    - lr_schedule (默认'kl_adaptive_minibatch'): 学习率调度策略，kl_adaptive_minibatch根据KL散度自适应调整
    
    【训练目标】
    - train_for_env_steps (默认1_000_000): 训练的总环境步数，实际训练中Follower约1e9，FollowerLite约2e7
    - save_best_metric (默认"avg_throughput"): 保存最佳模型的评估指标，训练中持续监控该指标
    
    【环境配置】
    - environment: 环境配置对象，包含地图、智能体、观测等环境相关参数
    - encoder: 编码器配置对象，定义神经网络编码器的架构
    - preprocessing: 预处理配置对象，定义观测预处理方式
    '''
    
    environment: EnvironmentMazes = EnvironmentMazes()  # 环境配置对象，包含地图、智能体、观测等环境相关参数
    encoder: EncoderConfig = EncoderConfig()  # 编码器配置对象，定义神经网络编码器的架构（ResNet层数、滤波器数量等）
    preprocessing: PreprocessorConfig = PreprocessorConfig()  # 预处理配置对象，定义观测预处理方式（动态/静态代价、网络输入半径等）

    rollout: int = 8  # 每个环境实例在收集数据前执行的最大步数，控制经验回放缓冲区大小，影响策略更新频率
    num_workers: int = 4  # 并行采样进程数，每个worker独立运行环境并收集数据，更多workers提高采样速度但增加资源消耗

    recurrence: int = 8  # RNN展开长度（训练时的序列长度），仅在use_rnn=True时有效，影响梯度传播距离和内存占用
    use_rnn: bool = False  # 是否使用RNN（循环神经网络），False使用MLP无记忆能力，True使用RNN/GRU处理时序依赖
    rnn_size: int = 256  # RNN隐藏层维度，仅在use_rnn=True时有效，更大维度增强表达能力但增加计算和内存

    ppo_clip_ratio: float = 0.1  # PPO裁剪比例（ε），限制策略更新幅度避免更新过大，裁剪范围[1-ε, 1+ε]，常用0.2
    batch_size: int = 2048  # 每次梯度更新的样本数量，从回放缓冲区采样batch_size个样本进行更新，常用16384

    exploration_loss_coeff: float = 0.018  # 探索损失（熵奖励）系数，鼓励策略探索避免过早收敛，值越大探索越强
    num_envs_per_worker: int = 4  # 每个worker并行运行的环境实例数，总并行环境数=num_workers×num_envs_per_worker
    worker_num_splits: int = 1  # worker进程是否拆分用于负载均衡，1表示不拆分，>1时将worker工作拆分到多个子进程
    max_policy_lag: int = 1  # 最大策略版本滞后数，采样worker使用的策略版本与当前策略版本的最大允许差距，1表示几乎同步

    force_envs_single_thread: bool = True  # 强制环境使用单线程，避免多线程环境与Sample Factory多进程冲突，提高稳定性
    optimizer: Literal["adam", "lamb"] = 'adam'  # 优化器类型，adam为Adam优化器，lamb为LAMB优化器适合大batch训练
    restart_behavior: str = "overwrite"  # 重启行为：["resume", "restart", "overwrite"]，overwrite表示覆盖之前的实验结果
    normalize_returns: bool = False  # 是否归一化回报（return），归一化可稳定训练但可能影响策略梯度估计
    async_rl: bool = False  # 是否使用异步强化学习，False为同步训练（标准PPO），True为异步训练（如IMPALA）
    num_batches_per_epoch: int = 16  # 每个训练epoch的批次数量，每个epoch使用num_batches_per_epoch×batch_size个样本

    num_batches_to_accumulate: int = 1  # 梯度累积的批次数，在更新前累积多个批次的梯度，用于模拟更大的batch size
    normalize_input: bool = False  # 是否归一化输入观测，归一化可提高训练稳定性但可能改变数据的原始分布
    decoder_mlp_layers = []  # 解码器MLP的隐藏层大小列表，空列表表示无额外的解码器层（使用默认架构）
    save_best_metric: str = "avg_throughput"  # 保存最佳模型的评估指标，训练中持续监控该指标保存最佳模型
    value_bootstrap: bool = True  # 是否使用价值函数引导，用价值函数估计未结束轨迹的未来奖励，提高回报估计准确性
    save_milestones_sec: int = -1  # 按时间间隔保存里程碑检查点（秒），-1表示禁用，>0时按时间间隔保存

    keep_checkpoints: int = 1  # 保留的检查点数量，仅保留最新的1个检查点，删除旧检查点以节省空间
    stats_avg: int = 10  # 统计指标的平滑窗口大小，使用最近10个回合的平均值进行统计，平滑波动
    learning_rate: float = 0.000146  # 学习率，控制参数更新步长，实际训练中Follower约0.00022，FollowerLite约0.00013
    train_for_env_steps: int = 1_000_000  # 训练的总环境步数，训练会持续直到累计环境交互步数达到该值

    gamma: float = 0.965  # 折扣因子（未来奖励的折扣），范围[0,1]越大越重视长期奖励，0.965表示未来第k步权重为0.965^k

    lr_schedule: str = 'kl_adaptive_minibatch'  # 学习率调度策略，kl_adaptive_minibatch根据KL散度自适应调整，constant为固定学习率

    experiment: str = 'exp'  # 实验名称标识符，用于区分不同的实验，会出现在保存路径和日志中
    train_dir: str = 'experiments/train_dir'  # 训练结果保存目录，包含检查点、日志、配置文件等
    seed: Optional[int] = 42  # 随机种子，用于确保实验可复现，相同种子会产生相同的随机数序列
    use_wandb: bool = True  # 是否使用wandb（Weights & Biases）进行实验跟踪和可视化，True时记录训练指标

    env: Literal['PogemaMazes-v0'] = "PogemaMazes-v0"  # 环境名称标识符，与Sample Factory框架中注册的环境名称对应
