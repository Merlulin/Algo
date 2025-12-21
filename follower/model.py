from argparse import Namespace
from typing import Literal

import torch
from pydantic import BaseModel
from sample_factory.model.encoder import Encoder
from sample_factory.utils.typing import Config, ObsSpace
from sample_factory.algo.utils.torch_utils import calc_num_elements

from sample_factory.utils.utils import log

from torch import nn as nn


class EncoderConfig(BaseModel):
    '''
    编码器配置类，定义ResNet编码器的架构参数。
    
    编码器的作用：
    - 将预处理后的观测（形状为(channels, height, width)的obs张量）转换为特征向量
    - 作为Actor-Critic网络的输入层，提取观测的空间特征
    - 在Sample Factory框架中，编码器输出的特征会传递给策略网络和价值网络
    
    参数说明：
    - num_filters: 卷积层的滤波器数量，决定特征图的通道数
      默认64，实际训练中Follower使用64，FollowerLite使用8
    - num_res_blocks: ResNet中残差块的数量，决定网络的深度
      默认1，实际训练中Follower使用8，FollowerLite使用1
    - activation_func: 激活函数类型，'ReLU'、'ELU'或'Mish'
      ReLU是最常用的，ELU在某些情况下性能更好，Mish是较新的激活函数
    - extra_fc_layers: 额外的全连接层数量（在卷积层之后）
      默认0表示不使用额外的全连接层
      实际训练中Follower使用1层（将卷积输出投影到hidden_size）
    - hidden_size: 额外全连接层的隐藏层维度
      默认128，实际训练中Follower使用512
      仅在extra_fc_layers > 0时使用
    
    实际训练配置示例：
    - Follower: num_filters=64, num_res_blocks=8, extra_fc_layers=1, hidden_size=512
    - FollowerLite: num_filters=8, num_res_blocks=1, extra_fc_layers=0
    '''
    encoder_arch: Literal['resnet', 'cnn_transformer', 'st_gat_former'] = 'cnn_transformer'
    extra_fc_layers: int = 0  # 额外全连接层数量，默认0（不使用）
    num_filters: int = 64  # 卷积层滤波器数量，决定特征图通道数
    num_res_blocks: int = 1  # ResNet残差块数量，决定网络深度
    activation_func: Literal['ReLU', 'ELU', 'Mish'] = 'ReLU'  # 激活函数类型
    hidden_size: int = 128  # 额外全连接层的隐藏层维度（仅在extra_fc_layers>0时使用）

    transformer_num_layers: int = 2
    transformer_nhead: int = 4
    transformer_dim_feedforward: int = 256
    transformer_dropout: float = 0.0
    transformer_use_cls_token: bool = False


def activation_func(cfg: EncoderConfig) -> nn.Module:
    '''
    激活函数工厂函数，根据配置返回对应的PyTorch激活函数模块。
    
    作用：
    - 根据EncoderConfig中的activation_func参数，返回对应的激活函数
    - 所有激活函数使用inplace=True，节省内存（直接修改输入而不创建新张量）
    
    支持的激活函数：
    - ReLU (Rectified Linear Unit): 最常用的激活函数，计算简单，梯度稳定
    - ELU (Exponential Linear Unit): 输出可以是负值，可能在某些情况下性能更好
    - Mish: 较新的平滑激活函数，在某些任务上表现优于ReLU
    
    参数:
        cfg: 编码器配置对象，包含activation_func参数
    
    返回:
        PyTorch激活函数模块（nn.Module）
    
    抛出异常:
        Exception: 如果配置中指定的激活函数未知
    '''
    if cfg.activation_func == "ELU":
        return nn.ELU(inplace=True)  # 指数线性单元
    elif cfg.activation_func == "ReLU":
        return nn.ReLU(inplace=True)  # 修正线性单元（最常用）
    elif cfg.activation_func == "Mish":
        return nn.Mish(inplace=True)  # Mish激活函数
    else:
        raise Exception("Unknown activation_func")  # 未知的激活函数类型


class ResBlock(nn.Module):
    '''
    ResNet残差块（Residual Block），ResNet编码器的核心组件。
    
    残差连接的作用：
    - 解决深度网络的梯度消失问题，使网络可以训练得更深
    - 通过恒等映射（identity mapping）允许梯度直接传播
    - 提高网络的表达能力和训练稳定性
    
    结构设计：
    - 标准残差块：激活 → 卷积 → 激活 → 卷积 → 残差连接
    - 使用3x3卷积核，stride=1，padding=1（保持空间尺寸不变）
    - 输入和输出的通道数相同（input_ch == output_ch），便于残差连接
    
    前向传播：
    - out = activation(conv(activation(conv(x)))) + x
    - 残差连接要求输入和输出形状相同
    
    参数:
        cfg: 编码器配置对象，用于确定激活函数类型
        input_ch: 输入通道数
        output_ch: 输出通道数（通常与input_ch相同）
    '''

    def __init__(self, cfg: EncoderConfig, input_ch, output_ch):
        '''
        初始化残差块，构建卷积层序列。
        
        构建的层序列：
        1. 激活函数
        2. 第一个3x3卷积（input_ch → output_ch）
        3. 激活函数
        4. 第二个3x3卷积（output_ch → output_ch，保持通道数）
        '''
        super().__init__()

        layers = [
            activation_func(cfg),  # 第一个激活函数
            nn.Conv2d(input_ch, output_ch, kernel_size=3, stride=1, padding=1),  # 第一个卷积层
            activation_func(cfg),  # 第二个激活函数
            nn.Conv2d(output_ch, output_ch, kernel_size=3, stride=1, padding=1),  # 第二个卷积层
            activation_func(cfg),  # 第三个激活函数
            nn.Conv2d(output_ch, output_ch, kernel_size=3, stride=1, padding=1),  # 第三个卷积层
        ]

        self.res_block_core = nn.Sequential(*layers)  # 残差块的核心层序列

    def forward(self, x):
        '''
        前向传播，实现残差连接。
        
        参数:
            x: 输入张量，形状为(batch, channels, height, width)
        
        返回:
            输出张量，形状与输入相同（通过残差连接保持）
        '''
        identity = x  # 保存输入作为残差连接的恒等映射
        out = self.res_block_core(x)  # 通过卷积层序列
        out = out + identity  # 残差连接：添加恒等映射
        return out


class ResnetEncoder(Encoder):
    '''
    ResNet编码器，将观测张量转换为特征向量，用于Actor-Critic网络。
    
    架构组成：
    1. 卷积头（conv_head）：
       - 初始卷积层：将输入通道数转换为num_filters
       - 多个残差块：提取空间特征（数量由num_res_blocks决定）
       - 最终激活函数
    2. 可选的全连接层（extra_linear）：
       - 如果extra_fc_layers > 0，添加全连接层进行特征投影
       - 将卷积输出的展平特征投影到hidden_size维度
    
    输入格式：
    - 观测字典，必须包含'obs'键
    - obs的形状：(channels, height, width)，其中channels由预处理阶段确定
      （如obstacles、agents等位置特征的数量）
    
    输出格式：
    - 特征向量，形状为(batch_size, encoder_out_size)
    - 可以直接输入到Actor-Critic网络的策略头和价值头
    
    在Sample Factory中的作用：
    - 作为自定义编码器，在register_training_utils.py中注册
    - Sample Factory框架会使用此编码器处理观测
    - 编码器输出会传递给RNN（如果use_rnn=True）或直接传递给策略/价值网络
    
    实际训练配置：
    - Follower: 8个残差块，64个滤波器，1层全连接（512维）
    - FollowerLite: 1个残差块，8个滤波器，无全连接层（更轻量）
    '''

    def __init__(self, cfg: Config, obs_space: ObsSpace):
        '''
        初始化ResNet编码器，构建网络架构。
        
        工作流程：
        1. 从配置中提取编码器配置
        2. 确定输入通道数（从obs_space['obs']的形状获取）
        3. 构建卷积头：初始卷积 + 残差块序列
        4. 计算卷积头输出大小
        5. 如果配置了extra_fc_layers，构建额外的全连接层
        
        参数:
            cfg: Sample Factory配置对象，包含encoder配置
            obs_space: 观测空间定义，用于确定输入形状
        '''
        super().__init__(cfg)
        self.encoder_cfg: EncoderConfig = EncoderConfig(**cfg.encoder)  # 解析编码器配置

        input_ch = obs_space['obs'].shape[0]  # 输入通道数（来自预处理阶段拼接的特征数）
        resnet_conf = [[self.encoder_cfg.num_filters, self.encoder_cfg.num_res_blocks]]  # 配置：[(输出通道数, 残差块数量)]
        curr_input_channels = input_ch
        layers = []

        # 构建卷积层和残差块
        for out_channels, res_blocks in resnet_conf:
            # 初始卷积层：将输入通道数转换为目标通道数
            layers.extend([nn.Conv2d(curr_input_channels, out_channels, kernel_size=3, stride=1, padding=1)])
            # 添加指定数量的残差块
            layers.extend([ResBlock(self.encoder_cfg, out_channels, out_channels) for _ in range(res_blocks)])
            curr_input_channels = out_channels

        layers.append(activation_func(self.encoder_cfg))  # 最终激活函数
        self.conv_head = nn.Sequential(*layers)  # 卷积头：所有卷积层和残差块的序列
        # 计算卷积头输出的展平后大小（用于确定全连接层输入维度）
        self.conv_head_out_size = calc_num_elements(self.conv_head, obs_space['obs'].shape)
        self.encoder_out_size = self.conv_head_out_size  # 编码器输出大小（如果无全连接层）

        # 如果配置了额外的全连接层，构建投影层
        if self.encoder_cfg.extra_fc_layers:
            self.extra_linear = nn.Sequential(
                nn.Linear(self.encoder_out_size, self.encoder_cfg.hidden_size),  # 线性投影
                activation_func(self.encoder_cfg),  # 激活函数
            )
            self.encoder_out_size = self.encoder_cfg.hidden_size  # 更新编码器输出大小

        log.debug('Convolutional layer output size: %r', self.conv_head_out_size)  # 记录卷积头输出大小

    def get_out_size(self) -> int:
        '''
        获取编码器输出特征的维度大小。
        
        返回:
            编码器输出特征向量的维度（用于构建后续网络层）
        
        使用场景：
        - Sample Factory框架使用此方法确定后续网络层的输入维度
        - Actor-Critic网络的策略头和价值头需要知道输入特征维度
        '''
        return self.encoder_out_size

    def forward(self, x):
        '''
        前向传播，将观测转换为特征向量。
        
        处理流程：
        1. 从输入字典中提取'obs'张量
        2. 通过卷积头提取空间特征
        3. 展平为特征向量
        4. 如果配置了额外全连接层，进行特征投影
        
        参数:
            x: 输入字典，必须包含'obs'键
               obs的形状：(batch_size, channels, height, width)
        
        返回:
            特征向量，形状为(batch_size, encoder_out_size)
            可以直接输入到后续的策略网络或价值网络
        '''
        x = x['obs']  # 提取观测张量
        x = self.conv_head(x)  # 通过卷积头（初始卷积 + 残差块序列 + 激活）
        x = x.contiguous().view(-1, self.conv_head_out_size)  # 展平为特征向量

        # 如果配置了额外全连接层，进行特征投影
        if self.encoder_cfg.extra_fc_layers:
            x = self.extra_linear(x)

        return x


class CNNTransformerEncoder(Encoder):
    def __init__(self, cfg: Config, obs_space: ObsSpace):
        super().__init__(cfg)
        self.encoder_cfg: EncoderConfig = EncoderConfig(**cfg.encoder)

        input_ch = obs_space['obs'].shape[0]
        h, w = obs_space['obs'].shape[1], obs_space['obs'].shape[2]
        self.num_tokens = int(h * w)

        d_model = int(self.encoder_cfg.num_filters)
        nhead = int(self.encoder_cfg.transformer_nhead)
        if d_model % nhead != 0:
            log.warning(
                'transformer_nhead (%d) does not divide d_model (%d), using nhead=1',
                nhead,
                d_model,
            )
            nhead = 1

        self.cnn = nn.Sequential(
            nn.Conv2d(input_ch, d_model, kernel_size=3, stride=1, padding=1),
            activation_func(self.encoder_cfg),
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=1, padding=1),
            activation_func(self.encoder_cfg),
        )

        seq_len = self.num_tokens + (1 if self.encoder_cfg.transformer_use_cls_token else 0)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

        if self.encoder_cfg.transformer_use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        else:
            self.cls_token = None

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=int(self.encoder_cfg.transformer_dim_feedforward),
            dropout=float(self.encoder_cfg.transformer_dropout),
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=int(self.encoder_cfg.transformer_num_layers))

        self.encoder_out_size = d_model
        if self.encoder_cfg.extra_fc_layers:
            self.extra_linear = nn.Sequential(
                nn.Linear(self.encoder_out_size, self.encoder_cfg.hidden_size),
                activation_func(self.encoder_cfg),
            )
            self.encoder_out_size = self.encoder_cfg.hidden_size

        log.debug('CNN-Transformer tokens: %r, d_model: %r', self.num_tokens, d_model)

    def get_out_size(self) -> int:
        return self.encoder_out_size

    def forward(self, x):
        x = x['obs']
        x = self.cnn(x)

        x = x.flatten(2).transpose(1, 2)
        if self.cls_token is not None:
            cls = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls, x], dim=1)

        x = x + self.pos_embed
        x = self.transformer(x)

        if self.cls_token is not None:
            x = x[:, 0]
        else:
            x = x.mean(dim=1)

        if self.encoder_cfg.extra_fc_layers:
            x = self.extra_linear(x)

        return x


def main():
    '''
    测试函数，用于验证ResNet编码器的创建和前向传播。
    
    功能：
    - 创建默认配置的编码器
    - 使用随机观测测试前向传播
    - 验证编码器是否能正常工作
    
    测试数据：
    - 观测半径r=5，对应11x11的观测窗口
    - 3个通道（模拟拼接后的位置特征）
    - 批量大小为1
    
    使用方式：
    - 直接运行此文件：python follower/model.py
    - 用于快速测试编码器是否正常工作
    - 调试编码器配置和前向传播逻辑
    '''
    exp_cfg = {'encoder': EncoderConfig().dict()}  # 创建默认编码器配置
    r = 5  # 观测半径
    obs = torch.rand(1, 3, r * 2 + 1, r * 2 + 1)  # 创建随机观测（batch=1, channels=3, 11x11）
    q_obs = {'obs': obs}  # 构建观测字典
    # noinspection PyTypeChecker
    re = ResnetEncoder(Namespace(**exp_cfg), dict(obs=obs[0]))  # 创建编码器（使用第一个样本的形状作为obs_space）
    re(q_obs)  # 前向传播测试


if __name__ == '__main__':
    main()
