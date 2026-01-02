import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common import Conv

class Concat(nn.Module):
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)

# BPS 
class BPS(nn.Module):
    def __init__(self, in_channels, out_channels, k=1, s=1): 
        super(BPS, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)

        # Apply stride 's' to the parallel branches to reduce computation resolution immediately
        self.d_cv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=s, padding=1,  dilation=1)
        self.d_cv2 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=s, padding=4,  dilation=2) 
        self.d_cv3 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=s, padding=12, dilation=4) 
        
        # Fusion layer uses kernel size 'k', stride 1 (since downsampling happened above)
        # Auto-padding: k // 2
        self.fusion = nn.Conv2d(3 * out_channels, out_channels, kernel_size=k, padding=k // 2)

        self.act = nn.SiLU()


    def forward(self, x):
        x_bn = self.bn(x) 
        
        out1 = self.d_cv1(x_bn)
        out2 = self.d_cv2(x_bn)
        out3 = self.d_cv3(x_bn)

        x_cat = torch.cat([out1, out2, out3], dim=1)

        out_fusion = self.fusion(x_cat)
        out_final = self.act(out_fusion)

        return out_final

# PMD-CFEM 
class PMD_CFEM(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(PMD_CFEM, self).__init__()

        # 如果未指定中间通道数，默认缩小一点以减少计算量
        if mid_channels is None:
            mid_channels = out_channels // 4

        # 1. 输入标准化 (Batch Normalization)
        # 放在最前面，类似 Pre-activation ResNet 的思路
        self.bn = nn.BatchNorm2d(in_channels)
        self.SiLU = nn.SiLU()

        # 2. 并行扩张卷积分支 (保持尺寸不变)
        self.d_cv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1,  dilation=1)
        self.d_cv2 = nn.Conv2d(in_channels, mid_channels, kernel_size=5, padding=4,  dilation=2) 
        self.d_cv3 = nn.Conv2d(in_channels, mid_channels, kernel_size=7, padding=12, dilation=4) 
        
        # 3. 融合层 (Fusion)
        # 输入通道 = 3个分支 * mid_channels
        self.fusion = nn.Conv2d(3 * mid_channels, out_channels, kernel_size=1)
        self.out_norm = nn.BatchNorm2d(out_channels) # 融合后的 BN

        # 4. 残差连接处理 (Skip Connection)
        # 如果输入通道 != 输出通道，或者输入尺寸变了（这里步长为1，尺寸应该不变），
        # 我们需要用 1x1 卷积调整输入 x 的形状，以便能够相加。
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.act = nn.SiLU()

    def forward(self, x):
        identity = x
        out = self.bn(x)
        out = self.SiLU(out)
        
        out1 = self.d_cv1(out)
        out2 = self.d_cv2(out)
        out3 = self.d_cv3(out)
        
        cat_out = torch.cat([out1, out2, out3], dim=1)
        out = self.fusion(cat_out)
        out = self.out_norm(out)
        
        out += self.shortcut(identity)
        
        return self.act(out)

class MP(nn.Module):
    def __init__(self, k=2):
        super(MP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return self.m(x)

#  MSDS-FM 
# 这个 Conv2d 两个通道参数一个其实是输入的维度，一个是卷积核数量
# 其实也就意思我对输入的数据要提取多少特征，那个 Conv_1x1 其实就是在不改变特征量的情况下做提纯
# 呃后面那个 group 参数如果等于输入的通道数就是深度卷积（后面得查一下前面写的代码有没有深度卷积）
# 还有那个 bias=False 是因为BN自带了所以这里不要了，前面可以回去补一下
# 激活函数这里用 SiLU 其实也有讲究（
# SiLU 计算量比较大，如果说你是想走那个啥轻量化那还是 SiLU 计算量少一点，而且其实SiLU自带了一点注意力机制
# 还有就是 SiLU 也不好训练，自身硬伤（梯度消失），这个是我在 YOLOv8 看到的，里面很多选用 SiLU
# 仔细一想有点头大，回头得改改（
# 现在很多函数名没有统一，跟咱前两天写的代码里的函数可能功能一样但是名字不一样（
# 下面的注释AI生成的，流程描述没啥问题
class MSDS_FM(nn.Module):
    def __init__(self, c1, c2):
        super(MSDS_FM, self).__init__()
        # assert c1 == c2, f"MSDS_FM input channels {c1} != output channels {c2}" # Removed for projection support
        channels = c1

        
        # -----------------------------------------------------------
        # 1. 定义三条并行的预处理支路
        # -----------------------------------------------------------
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        
        # 下路：Conv 1x1
        self.branch_1x1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU()
        )

        # -----------------------------------------------------------
        # 2. 中间处理模块
        # -----------------------------------------------------------
        
        # [C] 模块：拼接后降维
        self.concat_reduce = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU()
        )
        
        # [DWConv 1]：第一个深度卷积
        self.dw_conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU()
        )
        
        # [DWConv 2]：第二个深度卷积
        self.dw_conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU()
        )
        
        self.act = nn.SiLU()

        # 1x1 Conv projection if c1 != c2
        self.project = nn.Identity()
        if c1 != c2:
            self.project = nn.Sequential(
                nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(c2),
                nn.SiLU()
            )

    def forward(self, x):
        # --- 步骤 1: 三路分流 ---
        feat_avg = self.avg_pool(x)       # [路径A]: 原始背景特征
        feat_max = self.max_pool(x)
        feat_bottom = self.branch_1x1(x)
        
        # --- 步骤 2: 池化融合与第一层处理 ---
        # 拼接 -> 降维 -> DWConv1
        cat_out = torch.cat([feat_avg, feat_max], dim=1)
        unified_feat = self.concat_reduce(cat_out)
        
        # [关键输出]: DWConv1 的结果
        feat_dw1 = self.dw_conv1(unified_feat) # [路径B]: 浅层纹理特征
        
        # --- 步骤 3: 中间加法与第二层处理 ---
        # 中间混合：DWConv1 + 底部Conv1x1
        intermediate_sum = feat_dw1 + feat_bottom
        
        # DWConv2 处理
        feat_dw2 = self.dw_conv2(intermediate_sum) # [路径C]: 深层融合特征
        
        # --- 步骤 4: 最终三路融合 (修正点) ---
        # 图示逻辑：AvgPool结果 + DWConv1结果 + DWConv2结果
        final_out = feat_avg + feat_dw1 + feat_dw2
        
        # Apply Sigmoid then Projection (or Projection then Sigmoid? Typically Projection modifies channels)
        # Original: return self.Sigmoid(final_out)
        # We need to change channels from c1 to c2.
        
        return self.project(self.act(final_out))

# DDCA Module 
class DDCA(nn.Module):
    """
    Literal implementation of Dynamic Depthwise Channel Attention from your diagram.
    Uses three learnable weights (W1, W2, W3) to create the attention.
    """
    def __init__(self, c1, c2=None):
        super(DDCA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 1x1 Depthwise Conv as shown in the diagram
        self.dwconv = nn.Conv2d(c1, c1, kernel_size=1, stride=1, padding=0, groups=c1, bias=False)
        
        # The three learnable weights, initialized to 1
        self.w1 = nn.Parameter(torch.ones(1, c1, 1, 1))
        self.w2 = nn.Parameter(torch.ones(1, c1, 1, 1))
        self.w3 = nn.Parameter(torch.ones(1, c1, 1, 1))
        
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x_pooled is equivalent to W in the diagram
        x_pooled = self.avg_pool(x)
        # w_dyn is equivalent to W_dyn in the diagram
        w_dyn = self.dwconv(x_pooled)
        
        # Combine using the three learnable weights
        attention = self.w1 * x_pooled + self.w2 * w_dyn + self.w3 * (x_pooled * w_dyn)
        attention_map = self.Sigmoid(attention)
        
        return x * attention_map

# MS-PBSA Module 
class MS_PBSA(nn.Module):
    """
    Literal implementation of Multi-Scale Pooled-Bottleneck Spatial Attention.
    """
    def __init__(self, c1, c2=None, r=2): 
        super(MS_PBSA, self).__init__()
        if c2 is not None:
             assert c1 == c2, f"MS_PBSA input channels {c1} != output channels {c2}"
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        c_mid = max(16, c1 // r) # 建议：加一个 max 保护，防止通道数过小时变成 0
        
        self.path = nn.Sequential(
            # Step 1: 降维
            nn.Conv2d(c1, c_mid, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c_mid),
            nn.SiLU(),
            
            # Step 2: 深度卷积 (Depthwise Feature Extraction)
            nn.Conv2d(c_mid, c_mid, 3, 1, 1, groups=c_mid, bias=False),
            nn.BatchNorm2d(c_mid),
            nn.SiLU(),
            
            # Step 3: 升维 & 通道融合 (Generating Attention Logits)
            nn.Conv2d(c_mid, c1, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c1)
            # 注意：这里后面不加 SiLU，直接输出给 forward 里的 Sigmoid
        )
        
        # 核心修改：注意力机制必须用 Sigmoid 将权重限制在 [0, 1]
        self.Sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        _, _, H, W = x.shape
        
        # 1. Downsample
        s = self.pool(x)
        
        # 2. Bottleneck processing
        s = self.path(s)
        
        # 3. Upsample (Bilinear)
        # 将插值放在 Sigmoid 之前
        s = F.interpolate(s, size=(H, W), mode='bilinear', align_corners=False)
        
        # 4. Generate Weight Map (0~1)
        attention_map = self.Sigmoid(s)
        
        return x * attention_map


# SE Module
class SE(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# Cross Module 
class Cross(nn.Module):
    """
    Channel+Spatial Cross 模块的直译实现。
    该模块协调特征提取以及DDCA和MS-PBSA的应用，并按图示进行融合。
    """
    def __init__(self, c1, c2, k=1, s=1, g=1):
        super(Cross, self).__init__()
        c_mid = c1 // 2
        
        # 初始并行特征提取分支
        # self.dw_conv_3x3 = Conv(c1, c_mid, 3, 1, g=c_mid)
        self.dw_conv_3x3 = nn.Sequential(
            nn.Conv2d(c1, c_mid, kernel_size=3, stride=1, padding=1, groups=c_mid, bias=False),
            nn.BatchNorm2d(c_mid),
            nn.SiLU()
        )
        
        # self.dw_conv_5x5 = Conv(c1, c_mid, 5, 1, g=c_mid)
        self.dw_conv_5x5 = nn.Sequential(
            nn.Conv2d(c1, c_mid, kernel_size=5, stride=1, padding=2, groups=c_mid, bias=False),
            nn.BatchNorm2d(c_mid),
            nn.SiLU()
        )
        
        # 1x1 卷积用于在拼接后统一通道数
        # self.conv1x1 = Conv(c_mid * 2, c1, 1, 1)
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(c_mid * 2, c1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU()
        )
        
        # 图中所示的 SE 模块
        self.se = SE(c1)

        # 两个主要的注意力模块
        self.ddca = DDCA(c1)
        self.ms_pbsa = MS_PBSA(c1)
        
        # 用于融合的 GConv1x1，接受 2*c1 的拼接通道输入
        # 允许分组卷积，默认组为1（标准卷积）或用户指定的 g
        # 注意：GConv 的输入是 (Fc 拼接 Fs)，所以通道数 = c1 * 2。
        # 如果 g > 1，确保 g 能整除输入/输出通道数。
        # self.gconv = Conv(c1 * 2, c1, 1, 1, g=g)
        self.gconv = nn.Sequential(
            nn.Conv2d(c1 * 2, c1, kernel_size=1, stride=1, padding=0, groups=g, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU()
        )
        
        # 最终输出卷积以匹配 c2
        # self.out_conv = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()
        self.out_conv = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        ) if c1 != c2 else nn.Identity()

    def forward(self, x):
        # 初始特征提取和融合
        x1 = self.dw_conv_3x3(x)
        x2 = self.dw_conv_5x5(x)
        x_cat = torch.cat((x1, x2), dim=1)
        
        # x_fused 是图中残差连接的起始点
        x_fused = self.conv1x1(x_cat)
        
        # 应用 SE 模块
        x_se = self.se(x_fused)

        # 生成通道注意力 (Fc) 和空间注意力 (Fs) 特征
        fc = self.ddca(x_se)
        fs = self.ms_pbsa(x_se)
        
        # 拼接两个注意力特征
        fused_attention = torch.cat((fc, fs), dim=1)

        # 应用 GConv
        x_gconv = self.gconv(fused_attention)
        
        # 添加来自 x_fused (注意不是 x_se) 的残差连接
        out = x_fused + x_gconv
        
        return self.out_conv(out)
