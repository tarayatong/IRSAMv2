import torch
import os
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from dataset.image_floder import ImageFolder
from metrics import PD_FAmeter, mIoUmeter, nIoUmeter
from sam_spl.base_model import make_adaptor
from training import seed_everything

@torch.no_grad()
def evalution(test_loader, predictor, device, save_dir=None):
    predictor.eval()
    predictor = predictor.to(device)
    mIoU_meter = mIoUmeter()
    pdfa_meter = PD_FAmeter()
    nIoU_meter = nIoUmeter(1, 0.5)
    
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for _, (batch_data, gt_masks, image_names) in enumerate(test_loader):
        B, _, H, W = batch_data.shape
        gt_masks = gt_masks.to(device)
        batch_data = batch_data.to(device)
        pred_logit = predictor(batch_data)[0]
        pred_mask = pred_logit > 0
        
        if save_dir:
            pred_masks_np = (pred_mask.detach().cpu().squeeze(1).numpy() * 255).astype(np.uint8)
            for i in range(B):
                save_path = os.path.join(save_dir, image_names[i])
                Image.fromarray(pred_masks_np[i]).save(save_path)

        B, _, H, W = batch_data.shape

        nIoU_meter.update(
            pred_logit.reshape(B, 1, H, W).cpu().float(),
            gt_masks.reshape(B, 1, H, W).cpu().float(),
        )
        mIoU_meter.update(
            pred_mask.reshape(B, 1, H, W).cpu().float(),
            gt_masks.reshape(B, 1, H, W).cpu().float(),
        )
        pdfa_meter.update(
            pred_mask.reshape(B, H, W).cpu().float(),
            gt_masks.reshape(B, H, W).cpu().float(),
            [H, W],
        )
    return mIoU_meter, pdfa_meter, nIoU_meter


def main(
):
    # Argument parsing
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dataset", default="IRSTD-1k", type=str)
    parser.add_argument("--image_size", default=512, type=int)
    parser.add_argument("--batch_size", default=12, type=int)
    parser.add_argument("--data_path", default="./dataset/set_configs", type=str)
    parser.add_argument("--weights", default="./checkpoints/IRSTD-1k.pt", type=str)
    parser.add_argument("--device", default="cuda:1", type=str)
    parser.add_argument("--seed", default=111, type=int)
    parser.add_argument("--save_dir", default="./results/IRSTD-1k", type=str, help="Directory to save test results")
    FLAGS = parser.parse_args()

    # Settings
    dataset = FLAGS.dataset
    image_size = FLAGS.image_size
    batch_size = FLAGS.batch_size
    save_dir = FLAGS.save_dir
    seed_everything(FLAGS.seed)

    # load model
    predictor = make_adaptor(
                backbone_channel_list=[384, 192, 96],
                stages=[1, 2, 7],
                global_att_blocks=[5, 7, 9],
                block="res",
                embed_dim=96,
                dense_low_channels=[96, 48, 24],
                window_spec=(8, 4, 14),
                use_sam_decoder=True,
            )

    test_set = ImageFolder(FLAGS.data_path, dataset, istraining=False, base_size=image_size, crop_size=image_size)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=batch_size, pin_memory=True, num_workers=7)

    predictor_dict = torch.load(FLAGS.weights, map_location="cpu", weights_only=True)
    if "model_state_dict" in predictor_dict:
        predictor_dict = predictor_dict["model_state_dict"]
    predictor.load_state_dict(predictor_dict, strict=True)

    # Evaluation
    mIoU_meter, pdfa_meter, nIoU_meter = evalution(test_loader, predictor, FLAGS.device, save_dir)
    _, mIoU, fscore = mIoU_meter.get()
    nIoU = nIoU_meter.get()[1]
    pd, fa = pdfa_meter.get()
    result_line = f"mIoU = {mIoU*1e2: .2f} | F1Score = {fscore*1e2: .2f} | nIoU = {nIoU*1e2: .2f}  |  Pd = {pd*1e2:.2f}  | Fa = {fa*1e6:.2f}"

    print(result_line)
    
    if save_dir:
        with open(os.path.join(save_dir, "results.txt"), "a") as f:
            f.write(f"Dataset: {dataset}, Weights: {FLAGS.weights}\n")
            f.write(result_line + "\n")

if __name__ == "__main__":
    main()
