import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset.image_floder import ImageFolder
from sam_spl.base_model import make_adaptor
from training import seed_everything

def compute_distance(features, center):
    if features.shape[0] == 0:
        return np.array([])
    f_norm = F.normalize(features, p=2, dim=1)
    c_norm = F.normalize(center, p=2, dim=1)
    sim = (f_norm * c_norm).sum(dim=1)
    return (1.0 - sim).detach().cpu().numpy()

@torch.no_grad()
def visualize_features(test_loader, predictor, device, save_dir):
    predictor.eval()
    predictor = predictor.to(device)
    
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    all_dist_tgt = []
    all_dist_hf = []
    all_dist_lf = []

    for batch_idx, (batch_data, gt_masks, clutter_labels, image_names) in enumerate(tqdm(test_loader, desc="Visualizing Latent Space")):
        B, _, H, W = batch_data.shape
        gt_masks = gt_masks.to(device)
        batch_data = batch_data.to(device)
        clutter_labels = clutter_labels.to(device)
        
        # 前向传播，提取深层特征
        masks, return_dicts = predictor(batch_data)
        
        if return_dicts is None:
            print("Warning: return_dicts is None! 请确保 base_model.py 的 make_adaptor 中 use_alpha=True。")
            return
            
        # 在 return_dicts 中寻找 64 分辨率对应的特征层 
        target_embedding = None
        for r_dict in return_dicts:
            emb = r_dict["corrected_embedding"]
            # 自动定位 64x64 解析度的特征，如果没有则默认取最底层
            if emb.shape[-1] == 64 or emb.shape[-2] == 64:  
                target_embedding = emb
                break
                
        if target_embedding is None:
            # Fallback 如果没找到刚好 64的，拿第一个
            target_embedding = return_dicts[0]["corrected_embedding"]
            
        # (可选) 你想保留的 Mask 预测结果也可以存在这里
        pred_logit = masks[0][0]
        pred_mask = pred_logit > 0
        pred_masks_np = (pred_mask.detach().cpu().squeeze(1).numpy() * 255).astype(np.uint8)
        
        # 因为考虑到 Batch size, 我们对每个 batch element 循环
        for i in range(B):
            # 将预测面具保存下来
            img_prefix = os.path.splitext(image_names[i])[0]
            # Image.fromarray(pred_masks_np[i]).save(os.path.join(save_dir, f"{img_prefix}_pred.png"))
            
            emb = target_embedding[i:i+1] # [1, C, H_e, W_e]
            gt = gt_masks[i:i+1]          # [1, 1, H, W]
            clt = clutter_labels[i:i+1]   # [1, 1, H, W]
            
            _, C, He, We = emb.shape
            
            # 将 Label 降采样/对齐到 Embedding 分辨率
            if gt.shape[-2:] != (He, We):
                gt = F.interpolate(gt.float(), size=(He, We), mode='nearest')
            if clt.shape[-2:] != (He, We):
                clt = F.interpolate(clt.float(), size=(He, We), mode='nearest')
                
            emb_flat = emb.permute(0, 2, 3, 1).reshape(-1, C)
            tgt_flat = (gt.view(-1) > 0.5)
            clt_flat = (clt.view(-1) > 0.5)
            
            mask_target = tgt_flat
            mask_high_freq = clt_flat & (~tgt_flat)
            mask_low_freq = (~clt_flat) & (~tgt_flat)
            
            if not mask_target.any():
                continue # 没有目标的图片无法提供本组的参考质心
                
            emb_tgt = emb_flat[mask_target]
            emb_hf = emb_flat[mask_high_freq]
            emb_lf = emb_flat[mask_low_freq]
            
            centroid_tgt = emb_tgt.mean(dim=0, keepdim=True)
            
            dist_tgt = compute_distance(emb_tgt, centroid_tgt)
            dist_hf = compute_distance(emb_hf, centroid_tgt)
            dist_lf = compute_distance(emb_lf, centroid_tgt)
            
            # 随机抽样限制数量，防止最后画图崩溃
            sample_max = 256
            if len(dist_tgt) > sample_max: dist_tgt = np.random.choice(dist_tgt, sample_max, replace=False)
            if len(dist_hf) > sample_max: dist_hf = np.random.choice(dist_hf, sample_max, replace=False)
            if len(dist_lf) > sample_max: dist_lf = np.random.choice(dist_lf, sample_max, replace=False)
            
            all_dist_tgt.extend(dist_tgt)
            all_dist_hf.extend(dist_hf)
            all_dist_lf.extend(dist_lf)

    # ---------------- 汇聚整个测试集的数据开始画图 ----------------
    print(f"\nCollected Global Pixels - Target: {len(all_dist_tgt)}, High-freq: {len(all_dist_hf)}, Low-freq: {len(all_dist_lf)}.")
    if len(all_dist_tgt) == 0:
        print("未提取到任何有效目标点，无法绘图。")
        return

    data = []
    data.extend([{"Group": "Target vs. Target", "Distance": d} for d in all_dist_tgt])
    data.extend([{"Group": "Target vs. High-freq", "Distance": d} for d in all_dist_hf])
    data.extend([{"Group": "Target vs. Low-freq", "Distance": d} for d in all_dist_lf])
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(14, 9))
    sns.set_theme(style="whitegrid", palette="Set2") 
    
    ax = sns.violinplot(
        data=df, 
        x="Group", 
        y="Distance", 
        linewidth=2.0,
        inner="box",   
        cut=0
    )
    
    # plt.title("Latent Space Feature Distance to Target Centroid (NUAA-SIRST)", fontsize=48, fontweight='bold', pad=45)
    plt.xlabel("Pixel Groups", fontsize=42, fontweight='bold', labelpad=20)
    plt.ylabel(r"Cosine Distance ($1 - \cos\theta$)", fontsize=28, fontweight='bold', labelpad=20)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=36)
    
    plt.axhline(1.0, color='red', linestyle='--', linewidth=3.0, alpha=0.5, label='Orthogonal (1.0)')
    plt.legend(loc='upper right', fontsize=36)
    plt.ylim(-0.1, 2.1)
    
    plot_path = os.path.join(save_dir, "global_feature_distance_violin_ortho.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"============================================================")
    print(f"全局测试集 Violin Plot 渲染完毕并保存至: {plot_path}")
    print(f"============================================================")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dataset", default="NUAA-SIRST", type=str)
    parser.add_argument("--image_size", default=256, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--data_path", default="./dataset/set_configs", type=str)
    parser.add_argument("--weights", default=r"D:\workspace\paper\sam\IRSAMv2\checkpoints\last_nuaa_ortho.pt", type=str)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--seed", default=111, type=int)
    parser.add_argument("--save_dir", default="vis/visual_test", type=str, help="图表和分割结果保存目录")
    FLAGS = parser.parse_args()

    dataset = FLAGS.dataset
    image_size = FLAGS.image_size
    batch_size = FLAGS.batch_size
    save_dir = FLAGS.save_dir
    seed_everything(FLAGS.seed)

    # 确保设置 use_alpha=True，否则无法拿到 return_dicts 提取特征
    predictor = make_adaptor(
                backbone_channel_list=[384, 192, 96],
                stages=[1, 2, 7],
                global_att_blocks=[5, 7, 9],
                block="res",
                embed_dim=96,
                dense_low_channels=[96, 48, 24],
                window_spec=(8, 4, 14),
                use_sam_decoder=True,
                use_alpha=True,
            )

    test_set = ImageFolder(FLAGS.data_path, dataset, istraining=False, base_size=image_size, crop_size=image_size)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=batch_size, pin_memory=True, num_workers=7)

    predictor_dict = torch.load(FLAGS.weights, map_location="cpu", weights_only=True)
    if "model_state_dict" in predictor_dict:
        predictor_dict = predictor_dict["model_state_dict"]
    predictor.load_state_dict(predictor_dict, strict=False) # 设为 False 容忍一些新增修改的参数不匹配

    visualize_features(test_loader, predictor, FLAGS.device, save_dir)

if __name__ == "__main__":
    main()
