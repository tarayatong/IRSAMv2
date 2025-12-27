import torch
import torch.nn.functional as F

@torch.no_grad()
def compute_inverse_embedding(hyper_in_tokens, target_masks):
    """
    逆向计算 embedding：给定目标 mask，求解使得 token @ embedding = mask 的 embedding
    
    正向操作: masks = (hyper_in_tokens @ embedding.view(b, c, -1)).view(b, 1, h, w)
    逆向操作: embedding = hyper_in_tokens⁺ @ masks (求解)
    
    Args:
        hyper_in_tokens: [b, 1, c] - token向量，例如 (b, 1, 32)
        target_masks: [b, 1, h, w] - 目标mask
    
    Returns:
        unknown_embedding: [b, c, h, w] - 计算出的embedding
    """
    b, _, c = hyper_in_tokens.shape  # b, 1, 32
    _, _, h, w = target_masks.shape  # b, 1, h, w
    
    # 将 target_masks reshape 成 [b, 1, h*w]
    target_flat = target_masks.view(b, 1, h * w)
    
    # 计算伪逆: [b, 1, c] -> [b, c, 1]
    hyper_in_pinv = torch.linalg.pinv(hyper_in_tokens)
    
    # 矩阵乘法: [b, c, 1] @ [b, 1, h*w] = [b, c, h*w]
    unknown_embedding_flat = torch.bmm(hyper_in_pinv, target_flat)
    
    # reshape 回 [b, c, h, w]
    unknown_embedding = unknown_embedding_flat.view(b, c, h, w)
    
    return unknown_embedding

@torch.no_grad()
def AlphaLoss(out_dict, clutter_labels, target_labels, mode='geo'):
    """
    计算 Alpha Loss
    Args:
        out_dict: decoder 返回的字典，包含:
            - img_embedding: [b, 32, h, w]
            - edge_embeddings: [b, 32, h, w]
            - alpha: [b, 32, h, w]
            - hyper_in: [b, 1, 32]
        edges: 边缘图 [b, 1, h, w]
        labels_ori: 标签 [b, 1, h, w]，值范围 [0, 255]
        mode: 'geo' (几何正交), 'cos' (余弦相似度), 'mse' (均方误差)
    """
    # 从字典中获取需要的张量
    target_feat = out_dict.get("target_feat")  # [b, 32, h, w]
    clutter_feat = out_dict.get("clutter_feat")  # [b, 32, h, w]
    alpha = out_dict.get("alpha")  # [b, 32, h, w] 或 None
    corrected_embedding = out_dict.get("corrected_embedding")  # [b, 32, h, w]
    hyper_out = out_dict.get("hyper_out")  # [b, 1, 32] -> 需要 reshape to [b, 1, 32]
    
    # 确保 hyper_out 形状正确 [b, 1, c]
    if len(hyper_out.shape) == 2:
        hyper_out = hyper_out.unsqueeze(1)
    
    # 将 labels 归一化到 [0, 1] 并保持形状 [b, 1, h, w]
    y = target_labels / 255.0  # [B, 1, H, W]
    
    # 逆向计算理想的 embedding: 如果 hyper_tokens @ y_embedding = y，那么 y_embedding 是什么
    # y_embedding 的形状: [b, 32, h, w]
    y_embedding = compute_inverse_embedding(hyper_out, y)

    if target_feat.shape[-2:] != y_embedding.shape[-2:]:
        target_feat = F.interpolate(target_feat, size=y_embedding.shape[-2:], mode='bilinear', align_corners=False)
        clutter_feat = F.interpolate(clutter_feat, size=y_embedding.shape[-2:], mode='bilinear', align_corners=False)
        corrected_embedding = F.interpolate(corrected_embedding, size=y_embedding.shape[-2:], mode='bilinear', align_corners=False)

    p_min = target_feat.min(dim=1, keepdim=True)[0]  # [b, 1, h, w]
    p_max = target_feat.max(dim=1, keepdim=True)[0]  # [b, 1, h, w]
    p = (target_feat - p_min) / (p_max - p_min + 1e-8)  # [b, 32, h, w]

    q_min = clutter_feat.min(dim=1, keepdim=True)[0]  # [b, 1, h, w]
    q_max = clutter_feat.max(dim=1, keepdim=True)[0]  # [b, 1, h, w]
    q = (clutter_feat - q_min) / (q_max - q_min + 1e-8)  # [b, 32, h, w]

    p__min = corrected_embedding.min(dim=1, keepdim=True)[0]
    p__max = corrected_embedding.max(dim=1, keepdim=True)[0]
    p_ = (corrected_embedding - p__min) / (p__max - p__min + 1e-8)  # [b, 32, h, w]

    y_min = y_embedding.min(dim=1, keepdim=True)[0]
    y_max = y_embedding.max(dim=1, keepdim=True)[0]
    y_ = (y_embedding - y_min) / (y_max - y_min + 1e-8)

    if mode == 'cos':
        cos_sim = F.cosine_similarity(p_, y_, dim=1)  # [b, h, w]
        alpha_loss_val = (1 - cos_sim).mean()
        
    elif mode == 'geo':
        # 几何正交模式：要求 (y_embedding - alpha) ⊥ (masks - bgs)
        target1 = ((y_ - p_) * (p - q)).sum(dim=1, keepdim=True)  # [b, 1, h, w]
        alpha_loss_val = F.mse_loss(target1, torch.zeros_like(target1))
        
    else:
        if alpha is None:
            raise ValueError(f"mode='{mode}' 需要 alpha，但 out_dict 中 alpha 为 None")
        alpha_loss_val = F.mse_loss(p_, y_)

    return alpha_loss_val