"""
Loss functions for temporal FNO-CBAM
包含观测区域监督的损失函数
"""
import torch
import torch.nn.functional as F


def reconstruction_loss_missing(pred, target, missing_mask, ocean_mask):
    """
    重建损失：缺失区域（主要任务）

    Args:
        pred: (B, 1, H, W)
        target: (B, 1, H, W)
        missing_mask: (B, H, W) - 1=缺失, 0=观测
        ocean_mask: (B, H, W) - 1=海洋, 0=陆地

    Returns:
        loss
    """
    reconstruction_mask = (missing_mask * ocean_mask).unsqueeze(1)
    diff_squared = (pred - target) ** 2
    masked_diff = diff_squared * reconstruction_mask
    loss = masked_diff.sum() / (reconstruction_mask.sum() + 1e-8)
    return loss


def reconstruction_loss_observed(pred, target, missing_mask, ocean_mask):
    """
    重建损失：观测区域（辅助约束，防止破坏真实观测）

    Args:
        pred: (B, 1, H, W)
        target: (B, 1, H, W)
        missing_mask: (B, H, W) - 1=缺失, 0=观测
        ocean_mask: (B, H, W) - 1=海洋, 0=陆地

    Returns:
        loss
    """
    observed_mask = ((1 - missing_mask) * ocean_mask).unsqueeze(1)
    diff_squared = (pred - target) ** 2
    masked_diff = diff_squared * observed_mask
    loss = masked_diff.sum() / (observed_mask.sum() + 1e-8)
    return loss


def gradient_loss(pred, target, missing_mask, ocean_mask):
    """
    梯度匹配损失：减少锋面位置偏移导致的大单点误差

    只在缺失区域计算
    """
    reconstruction_mask = (missing_mask * ocean_mask).unsqueeze(1)

    # Y方向梯度
    pred_grad_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_grad_y = target[:, :, 1:, :] - target[:, :, :-1, :]
    mask_grad_y = reconstruction_mask[:, :, 1:, :]

    # X方向梯度
    pred_grad_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    target_grad_x = target[:, :, :, 1:] - target[:, :, :, :-1]
    mask_grad_x = reconstruction_mask[:, :, :, 1:]

    # 计算梯度误差
    grad_loss_y = torch.abs(pred_grad_y - target_grad_y) * mask_grad_y
    grad_loss_x = torch.abs(pred_grad_x - target_grad_x) * mask_grad_x

    total_grad_loss = (grad_loss_y.sum() / (mask_grad_y.sum() + 1e-8) +
                       grad_loss_x.sum() / (mask_grad_x.sum() + 1e-8)) / 2

    return total_grad_loss


def laplacian_loss(pred, target, missing_mask, ocean_mask):
    """
    拉普拉斯损失（PDE约束）：二阶空间导数匹配

    海温场应满足扩散方程的空间平滑性约束：
    ∇²T_pred ≈ ∇²T_target

    拉普拉斯算子: ∇²T = ∂²T/∂x² + ∂²T/∂y²

    Args:
        pred: (B, 1, H, W)
        target: (B, 1, H, W)
        missing_mask: (B, H, W) - 1=缺失, 0=观测
        ocean_mask: (B, H, W) - 1=海洋, 0=陆地

    Returns:
        loss
    """
    reconstruction_mask = (missing_mask * ocean_mask).unsqueeze(1)

    # 拉普拉斯算子卷积核 (二阶导数)
    # [0,  1, 0]
    # [1, -4, 1]
    # [0,  1, 0]
    laplacian_kernel = torch.tensor(
        [[[[0, 1, 0],
           [1, -4, 1],
           [0, 1, 0]]]],
        dtype=pred.dtype, device=pred.device
    )

    # 计算拉普拉斯
    pred_laplacian = F.conv2d(pred, laplacian_kernel, padding=1)
    target_laplacian = F.conv2d(target, laplacian_kernel, padding=1)

    # 在缺失区域计算拉普拉斯匹配损失
    laplacian_diff = torch.abs(pred_laplacian - target_laplacian)
    masked_laplacian = laplacian_diff * reconstruction_mask

    loss = masked_laplacian.sum() / (reconstruction_mask.sum() + 1e-8)

    return loss


def laplacian_smoothness_loss(pred, missing_mask, ocean_mask):
    """
    拉普拉斯平滑损失（无需target）：鼓励预测场的空间平滑性

    最小化 |∇²T_pred|，即预测场应该是平滑的

    Args:
        pred: (B, 1, H, W)
        missing_mask: (B, H, W) - 1=缺失, 0=观测
        ocean_mask: (B, H, W) - 1=海洋, 0=陆地

    Returns:
        loss
    """
    reconstruction_mask = (missing_mask * ocean_mask).unsqueeze(1)

    # 拉普拉斯算子
    laplacian_kernel = torch.tensor(
        [[[[0, 1, 0],
           [1, -4, 1],
           [0, 1, 0]]]],
        dtype=pred.dtype, device=pred.device
    )

    pred_laplacian = F.conv2d(pred, laplacian_kernel, padding=1)

    # 最小化拉普拉斯幅度（平滑约束）
    smoothness = torch.abs(pred_laplacian) * reconstruction_mask

    loss = smoothness.sum() / (reconstruction_mask.sum() + 1e-8)

    return loss


def temporal_consistency_loss_linear(pred, sst_seq, ocean_mask, lookback=10, threshold=2.0):
    """
    时间连续性约束（方法1）：多点线性回归外推

    使用最近lookback天的数据进行线性回归，预测第30天的趋势

    Args:
        pred: (B, 1, H, W) - 第30天预测
        sst_seq: (B, 30, H, W) - 30天序列
        ocean_mask: (B, H, W)
        lookback: 用于回归的天数（默认10天）
        threshold: 允许偏离趋势的阈值（°C）
    """
    B, T, H, W = sst_seq.shape

    # 取最近lookback天的数据
    recent_seq = sst_seq[:, -lookback:, :, :]  # (B, lookback, H, W)

    # 时间序列: 0, 1, 2, ..., lookback-1
    t = torch.arange(lookback, dtype=torch.float32, device=pred.device)  # (lookback,)

    # 计算线性回归系数
    # y = a*t + b
    # a = (n*Σ(t*y) - Σt*Σy) / (n*Σt² - (Σt)²)
    # b = (Σy - a*Σt) / n

    # Reshape: (B, lookback, H, W) → (B, H, W, lookback)
    recent_seq = recent_seq.permute(0, 2, 3, 1)  # (B, H, W, lookback)

    n = float(lookback)
    sum_t = t.sum()
    sum_t2 = (t ** 2).sum()
    sum_y = recent_seq.sum(dim=-1)  # (B, H, W)
    sum_ty = (recent_seq * t.view(1, 1, 1, -1)).sum(dim=-1)  # (B, H, W)

    # 计算斜率a
    a = (n * sum_ty - sum_t * sum_y) / (n * sum_t2 - sum_t ** 2 + 1e-8)
    # 计算截距b
    b = (sum_y - a * sum_t) / n

    # 外推到第30天（t = lookback）
    expected = a * lookback + b  # (B, H, W)
    expected = expected.unsqueeze(1)  # (B, 1, H, W)

    # 计算偏差
    threshold_norm = threshold / 2.69  # 归一化阈值
    deviation = torch.abs(pred - expected)
    penalty = F.relu(deviation - threshold_norm)

    # 只在海洋区域计算
    mask = ocean_mask.unsqueeze(1)
    loss = (penalty * mask).sum() / (mask.sum() + 1e-8)

    return loss


def temporal_consistency_loss_stats(pred, sst_seq, ocean_mask):
    """
    时间连续性约束（方法2）：统计变化约束

    分析前29天的日间变化统计特性，确保第30天的变化在合理范围内

    Args:
        pred: (B, 1, H, W) - 第30天预测
        sst_seq: (B, 30, H, W) - 30天序列
        ocean_mask: (B, H, W)
    """
    # 计算前29天的日间变化
    daily_changes = sst_seq[:, 1:, :, :] - sst_seq[:, :-1, :, :]  # (B, 29, H, W)

    # 统计量
    mean_change = daily_changes.mean(dim=1)  # (B, H, W)
    std_change = daily_changes.std(dim=1)  # (B, H, W)

    # 第30天相对第29天的预测变化
    last_day = sst_seq[:, -1:, :, :]  # (B, 1, H, W)
    pred_change = pred - last_day  # (B, 1, H, W)

    # 标准化偏差：超过3个标准差视为异常
    deviation = torch.abs(pred_change.squeeze(1) - mean_change) / (std_change + 1e-8)
    penalty = F.relu(deviation - 3.0)  # 只惩罚超过3σ的部分

    # 只在海洋区域计算
    mask = ocean_mask
    loss = (penalty * mask).sum() / (mask.sum() + 1e-8)

    return loss


def temporal_consistency_loss_accel(pred, sst_seq, ocean_mask, window=5):
    """
    时间连续性约束（方法3）：加速度平滑约束（物理驱动）

    海温变化应该是平滑的，加速度（二阶导数）不应突变

    Args:
        pred: (B, 1, H, W) - 第30天预测
        sst_seq: (B, 30, H, W) - 30天序列
        ocean_mask: (B, H, W)
        window: 用于计算加速度的窗口（默认5天）
    """
    # 取最近window天
    recent_seq = sst_seq[:, -window:, :, :]  # (B, window, H, W)

    # 一阶差分（速度）
    velocity = recent_seq[:, 1:, :, :] - recent_seq[:, :-1, :, :]  # (B, window-1, H, W)

    # 二阶差分（加速度）
    acceleration = velocity[:, 1:, :, :] - velocity[:, :-1, :, :]  # (B, window-2, H, W)

    # 预测第30天的速度
    last_velocity = sst_seq[:, -1:, :, :] - sst_seq[:, -2:-1, :, :]  # (B, 1, H, W)
    pred_velocity = pred - sst_seq[:, -1:, :, :]  # (B, 1, H, W)

    # 预测的加速度
    pred_accel = pred_velocity - last_velocity  # (B, 1, H, W)

    # 历史加速度的平均幅度
    historical_accel_magnitude = torch.abs(acceleration).mean(dim=1, keepdim=True)  # (B, 1, H, W)

    # 惩罚过大的加速度变化（超过历史均值的2倍）
    penalty = F.relu(torch.abs(pred_accel) - 2.0 * historical_accel_magnitude)

    # 只在海洋区域计算
    mask = ocean_mask.unsqueeze(1)
    loss = (penalty * mask).sum() / (mask.sum() + 1e-8)

    return loss


def temporal_consistency_loss_multi(pred, sst_seq, ocean_mask):
    """
    时间连续性约束（组合方法）：结合线性回归、统计约束和加速度平滑

    Args:
        pred: (B, 1, H, W) - 第30天预测
        sst_seq: (B, 30, H, W) - 30天序列
        ocean_mask: (B, H, W)
    """
    # 三种约束
    loss_linear = temporal_consistency_loss_linear(pred, sst_seq, ocean_mask, lookback=10)
    loss_stats = temporal_consistency_loss_stats(pred, sst_seq, ocean_mask)
    loss_accel = temporal_consistency_loss_accel(pred, sst_seq, ocean_mask, window=5)

    # 加权组合：线性回归和统计约束为主，加速度为辅
    total_loss = 0.4 * loss_linear + 0.4 * loss_stats + 0.2 * loss_accel

    return total_loss


def temporal_consistency_loss(pred, sst_seq, ocean_mask, threshold=2.0):
    """
    时间连续性约束：预测应该延续历史趋势（旧版本，保留向后兼容）

    Args:
        pred: (B, 1, H, W) - 第30天预测
        sst_seq: (B, 30, H, W) - 30天序列（包含目标日）
        ocean_mask: (B, H, W)
        threshold: 允许偏离趋势的阈值（归一化后的值，默认~2°C/2.69）

    注意：只使用前29天的数据，不能用第30天的输入（避免信息泄露）
    """
    # 使用新的组合方法
    return temporal_consistency_loss_multi(pred, sst_seq, ocean_mask)


def physical_range_constraint(pred, ocean_mask, mean=26.71, std=2.69):
    """
    物理温度范围约束：防止非物理预测

    Args:
        pred: (B, 1, H, W) - 归一化后的预测
        ocean_mask: (B, H, W)
        mean, std: 归一化参数
    """
    # 物理范围（反归一化到真实温度）
    T_min_real, T_max_real = -2.0, 35.0

    # 转换到归一化空间
    T_min_norm = (T_min_real - mean) / std
    T_max_norm = (T_max_real - mean) / std

    # 计算超出范围的惩罚
    penalty_low = F.relu(T_min_norm - pred) ** 2
    penalty_high = F.relu(pred - T_max_norm) ** 2

    mask = ocean_mask.unsqueeze(1)
    loss = ((penalty_low + penalty_high) * mask).sum() / (mask.sum() + 1e-8)

    return loss


def combined_loss_temporal(pred, target, missing_mask, ocean_mask,
                           sst_seq=None,
                           alpha_missing=1.0, alpha_observed=0.1, gamma=0.15,
                           beta_temporal=0.15, beta_range=0.01,
                           beta_laplacian=0.0):
    """
    时序FNO-CBAM的组合损失（含物理约束）

    Args:
        pred: (B, 1, H, W)
        target: (B, 1, H, W)
        missing_mask: (B, H, W)
        ocean_mask: (B, H, W)
        sst_seq: (B, 30, H, W) - 30天历史序列（用于时间连续性约束）
        alpha_missing: 缺失区域权重 (默认1.0)
        alpha_observed: 观测区域权重 (默认0.1，如果使用输出组合可设为0)
        gamma: 梯度loss权重 (默认0.15)
        beta_temporal: 时间连续性权重 (默认0.15)
        beta_range: 温度范围约束权重 (默认0.01)
        beta_laplacian: 拉普拉斯PDE约束权重 (默认0.0)

    Returns:
        total_loss, loss_missing, loss_observed, loss_grad
    """
    # 缺失区域loss（主要任务）
    loss_missing = reconstruction_loss_missing(pred, target, missing_mask, ocean_mask)

    # 观测区域loss（防止破坏特征，如果使用输出组合则可设为0）
    if alpha_observed > 0:
        loss_observed = reconstruction_loss_observed(pred, target, missing_mask, ocean_mask)
    else:
        loss_observed = torch.tensor(0.0, device=pred.device)

    # 梯度loss（保持物理特征 - 一阶PDE）
    if gamma > 0:
        loss_grad = gradient_loss(pred, target, missing_mask, ocean_mask)
    else:
        loss_grad = torch.tensor(0.0, device=pred.device)

    # PDE约束：拉普拉斯（二阶空间导数匹配）
    if beta_laplacian > 0:
        loss_laplacian = laplacian_loss(pred, target, missing_mask, ocean_mask)
    else:
        loss_laplacian = torch.tensor(0.0, device=pred.device)

    # 物理约束1：时间连续性
    if sst_seq is not None and beta_temporal > 0:
        loss_temporal = temporal_consistency_loss(pred, sst_seq, ocean_mask)
    else:
        loss_temporal = torch.tensor(0.0, device=pred.device)

    # 物理约束2：温度范围
    if beta_range > 0:
        loss_range = physical_range_constraint(pred, ocean_mask)
    else:
        loss_range = torch.tensor(0.0, device=pred.device)

    # 总loss
    total_loss = (alpha_missing * loss_missing +
                  alpha_observed * loss_observed +
                  gamma * loss_grad +
                  beta_laplacian * loss_laplacian +
                  beta_temporal * loss_temporal +
                  beta_range * loss_range)

    return total_loss, loss_missing, loss_observed, loss_grad


if __name__ == '__main__':
    # 测试
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    B, H, W = 4, 451, 351
    pred = torch.randn(B, 1, H, W).to(device)
    target = torch.randn(B, 1, H, W).to(device)
    missing_mask = torch.randint(0, 2, (B, H, W)).float().to(device)
    ocean_mask = torch.ones(B, H, W).to(device)

    print("="*60)
    print("Loss函数测试")
    print("="*60)

    total_loss, loss_missing, loss_observed, loss_grad = combined_loss_temporal(
        pred, target, missing_mask, ocean_mask
    )

    print(f"\nLoss值:")
    print(f"  Total: {total_loss.item():.6f}")
    print(f"  Missing (α=1.0): {loss_missing.item():.6f}")
    print(f"  Observed (α=0.1): {loss_observed.item():.6f}")
    print(f"  Gradient (γ=0.15): {loss_grad.item():.6f}")

    print("\n" + "="*60)
    print("✓ Loss函数测试通过")
