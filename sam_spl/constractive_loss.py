import torch
import torch.nn.functional as F

def compute_target_centric_contrastive_loss(emb_prime, w_t, gt_mask, margin_push=0.9):
    """
    Args:
        emb_prime: 修正后的 Embedding [B, C, H, W] (return_dict["corrected_embedding"])
        w_t: 目标查询向量 [B, C] (return_dict["w_t"])
        gt_mask: 目标 Ground Truth [B, 1, H, W] (batch_masks)
        margin_push: 难样本推开的阈值 (eta)
    """
    B, C, H, W = emb_prime.shape

    # 1. 归一化，准备计算余弦相似度
    # emb_norm: [B, C, H, W], w_t_norm: [B, C]
    emb_norm = F.normalize(emb_prime, p=2, dim=1)
    w_t_norm = F.normalize(w_t.detach(), p=2, dim=1)
    
    # 2. 计算每个像素与目标方向的相似度图 (Similarity Map)
    # [B, C, H, W] * [B, C, 1, 1] -> 沿通道求和 -> [B, H, W]
    sim_map = torch.einsum('bchw,bc->bhw', emb_norm, w_t_norm)
    
    if gt_mask.shape != (B, 1, H, W):
        sim_map = F.interpolate(sim_map.unsqueeze(1), size=gt_mask.shape[2:], 
                             mode='bilinear', align_corners=False).squeeze(1)
    pos_mask = (gt_mask > 0.5).squeeze(1) # [B, H, W]
    if pos_mask.any():
        # 让目标的相似度趋近于 1
        loss_pull = (1.0 - sim_map[pos_mask]).mean()
    else:
        loss_pull = torch.tensor(0.0, device=emb_prime.device)
        
    # 4. 难样本推开 (Hard Negative Pushing)
    # 提取 GT 为 0 的像素 (背景)
    neg_mask = (gt_mask < 0.5).squeeze(1) # [B, H, W]
    if neg_mask.any():
        # 只有当背景像素的相似度超过 margin 时，才产生惩罚
        # 让背景的相似度趋近于 0 或负数
        loss_push = torch.clamp(sim_map[neg_mask] - margin_push, min=0).mean()
    else:
        loss_push = torch.tensor(0.0, device=emb_prime.device)
        
    return loss_pull, loss_push


def compute_relative_contrastive_loss(emb_prime, w_t, w_c, gt_mask, temperature=0.1):
    """
    修改后：相对原型对比损失 
    利用目标原型(w_t)和杂波原型(w_c)作为一个局部的二分类极点，代替暴力的硬阈值推开措施。
    
    Args:
        emb_prime: 修正后的 Embedding [B, C, H, W] 
        w_t: 目标查询向量 [B, C] 
        w_c: 负向/干扰查询向量 [B, C] (可以传入你前向中正交化后的 return_dict["w_c_ir"])
        gt_mask: 目标 Ground Truth [B, 1, H, W]
        temperature: 温度缩放系数(与你的 \tau 一致也可，或者 0.07~0.2)。用于控制惩罚的敏锐程度
    """
    B, C, H, W = emb_prime.shape
    # 1. 全部转入同一超求面上
    emb_norm = F.normalize(emb_prime, p=2, dim=1)
    w_t_norm = F.normalize(w_t.detach(), p=2, dim=1)
    w_c_norm = F.normalize(w_c.detach(), p=2, dim=1) 
    
    # 2. 分别计算与 目标(t) 和 杂波/背景(c) 的相似度并用温度放大动态敏感性区间
    # [b, c, h, w] * [b, c] -> [b, h, w]
    sim_t = torch.einsum('bchw,bc->bhw', emb_norm, w_t_norm) / temperature
    sim_c = torch.einsum('bchw,bc->bhw', emb_norm, w_c_norm) / temperature
    
    # 3. 组成每个像元的逻辑层 (Logits) : [B, 2, H, W]
    # 通道索引0代表 '归类为背景原型' 的分数 ， 1 代表 '归类为目标原型' 的分数
    logits = torch.stack([sim_c, sim_t], dim=1) 
    
    # 4. 对齐 GT 并变成类别编号 (0,1)
    if gt_mask.shape != (B, 1, H, W):
        gt_mask = F.interpolate(gt_mask.float(), size=(H, W), mode='nearest')
    
    # 背景为 0, 前景为 1 [B, H, W]
    labels = (gt_mask > 0.5).long().squeeze(1)
    # 5. 直接过交叉熵！
    # 交叉熵内部相当于运行 softmax(sim_t, sim_c)。
    # 对于那些具有相似语义、比较难分类的背景像素，由于有 w_c 这个安全岛兜底，
    # 只要它的 sim_c 开始大过 sim_t，Loss 梯度就会进入平缓的指数退火曲线。彻底消灭因为 0.1 指标强拉产生的震荡。
    loss = F.cross_entropy(logits, labels)
    
    return loss


def compute_smoothed_target_contrastive_loss(emb_prime, w_t, gt_mask, temperature=0.1):
    B, C, H, W = emb_prime.shape
    emb_norm = F.normalize(emb_prime, p=2, dim=1)
    w_t_norm = F.normalize(w_t.detach(), p=2, dim=1)
    sim_map = torch.einsum('bchw,bc->bhw', emb_norm, w_t_norm)
    
    if gt_mask.shape != (B, 1, H, W):
        gt_mask = F.interpolate(gt_mask.float(), size=(H, W), mode='nearest')
    
    pos_mask = (gt_mask > 0.5).float().squeeze(1)
    
    # 利用类似 tau 的思路引入温度并拉直进入 Sigmoid，
    # 这样负样本被 push 的时候，用的是 log 指数软落地，而不是 hard 的撞墙。
    sim_scaled = sim_map / temperature
    # reduction="mean" 直接接管
    loss = F.binary_cross_entropy_with_logits(sim_scaled, pos_mask)
    
    return loss


def compute_manifold_contrastive_loss(emb_prime, gt_mask, temperature=0.1):
    B, C, H, W = emb_prime.shape
    if gt_mask.shape != (B, 1, H, W):
        gt_mask = F.interpolate(gt_mask.float(), size=(H, W), mode='nearest')
    emb_flat = emb_prime.view(B, C, -1)     
    mask_flat = (gt_mask > 0.5).float().view(B, 1, -1) 
    eps = 1e-6
    # 动态求取真实的特征大本营（质心）
    fg_center = (emb_flat * mask_flat).sum(dim=-1, keepdim=True) / (mask_flat.sum(dim=-1, keepdim=True) + eps)
    bg_center = (emb_flat * (1.0 - mask_flat)).sum(dim=-1, keepdim=True) / ((1.0 - mask_flat).sum(dim=-1, keepdim=True) + eps)
    
    emb_norm = F.normalize(emb_flat, p=2, dim=1) 
    fg_center_norm = F.normalize(fg_center, p=2, dim=1) 
    bg_center_norm = F.normalize(bg_center, p=2, dim=1) 
    
    # 构建 Logits 并用无参数的 CrossEntropy 算平滑聚类损失
    sim_fg = (emb_norm * fg_center_norm).sum(dim=1) / temperature
    sim_bg = (emb_norm * bg_center_norm).sum(dim=1) / temperature
    logits = torch.stack([sim_bg, sim_fg], dim=1)
    
    loss_contrastive = F.cross_entropy(logits, mask_flat.squeeze(1).long())
    return loss_contrastive


def compute_ohem_bce_loss(pred_logits, gt_mask):
    """
    带有高频杂波镇压机制的自适应 BCE Loss。
    pred_logits: 乘过了 tau 的预测图
    """
    p_pred = torch.sigmoid(pred_logits) # 预测概率
    loss_bce = F.binary_cross_entropy_with_logits(pred_logits, gt_mask.float(), reduction='none')
    
    weight_map = torch.ones_like(p_pred)
    neg_mask = (gt_mask <= 0.5)
    
    # OHEM 核心：对于背景像素，它被预测为目标的概率越高（越像目标的杂波），分配给它的惩罚权重就越大！
    # 垫底加个 0.1 保证最平滑的背景也有微弱更新
    weight_map[neg_mask] = p_pred[neg_mask].detach() + 0.1 
    
    return (loss_bce * weight_map).mean()
