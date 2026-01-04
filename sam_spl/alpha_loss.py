import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

@torch.no_grad()
def AlphaLoss(out_dict, clutter_labels, target_labels, mode='geo'):

    target_feat = out_dict.get("target_feat")  # [b, 32, h, w]
    clutter_feat = out_dict.get("clutter_feat")  # [b, 32, h, w]
    corrected_embedding = out_dict.get("corrected_embedding")  # [b, 32, h, w]

    ortho_loss_val = ((corrected_embedding - target_feat) * (target_feat - clutter_feat)).sum(dim=1).pow(2).mean()  # [b, 1, h, w]
    L_energy = corrected_embedding.pow(2).mean()

    return ortho_loss_val+L_energy