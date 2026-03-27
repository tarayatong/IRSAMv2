import torch
import torch.nn.functional as F

def compute_target_centric_contrastive_loss(emb_prime, w_t, gt_mask, margin_push=0.1):
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