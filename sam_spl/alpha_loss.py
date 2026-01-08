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
    w_t = out_dict.get("w_t")
    w_c = out_dict.get("w_c")
    corrected_embedding = out_dict.get("corrected_embedding")
    p_ = F.normalize(corrected_embedding, dim=1, eps=1e-6).sum(dim=1, keepdim=True)
    if p_.shape[-2:] != target_labels.shape[-2:]:
        p_ = F.interpolate(p_, size=target_labels.shape[-2:], mode='bilinear', align_corners=False)
    alpha_loss_val = (w_t* w_c).sum(dim=1).mean() + 0.1*F.mse_loss(p_, target_labels)

    return alpha_loss_val