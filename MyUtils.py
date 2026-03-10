import math


def get_lr_lambda(epoch):
    warmup_epochs = 10
    total_epochs = 30
    eta_min_ratio = 0.01  # 最小学习率占初始学习率的比例
    
    # 1. Warmup 阶段：线性增加
    if epoch < warmup_epochs:
        return float(epoch) / float(max(1, warmup_epochs))
    
    # 2. Cosine 阶段
    # 计算在余弦阶段走了多远 (0 ~ 1 之间)
    progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
    # 余弦公式实现
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    
    # 结合最小 LR 比例，防止降为 0
    return eta_min_ratio + (1.0 - eta_min_ratio) * cosine_factor

def get_lr_lambda2(epoch):
    warmup_epochs = 5
    total_epochs = 30

    # 1. Warmup 阶段：线性增加
    if epoch < warmup_epochs:
        return float(epoch) / float(max(1, warmup_epochs)) * 3
    elif epoch < 10:
        return 3
    elif epoch < 15:
        return 2
    elif epoch < 20:
        return 1.5
    elif epoch < 25:
        return 1
    else:
        return 0.5
