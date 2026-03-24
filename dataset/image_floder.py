import os
import random
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import Dataset


def Normalized(img, dataset):
    if dataset == "NUDT-SIRST":
        img = (img - 107.80905151367188) / 33.02274703979492
    elif dataset == "NUAA-SIRST":
        img = (img - 101.06385040283203) / 34.619606018066406
    elif dataset == "IRSTD-1k":
        img = (img - 87.4661865234375) / 39.71953201293945
    return img


def random_crop(img, mask, patch_size, pos_prob=None):
    h, w, c = img.shape
    if min(h, w) < patch_size:
        img = np.pad(
            img,
            ((0, max(h, patch_size) - h), (0, max(w, patch_size) - w), (0, 0)),
            mode="constant",
        )
        mask = np.pad(
            mask,
            ((0, max(h, patch_size) - h), (0, max(w, patch_size) - w)),
            mode="constant",
        )
        h, w, c = img.shape

    while 1:
        h_start = random.randint(0, h - patch_size)
        h_end = h_start + patch_size
        w_start = random.randint(0, w - patch_size)
        w_end = w_start + patch_size

        img_patch = img[h_start:h_end, w_start:w_end]
        mask_patch = mask[h_start:h_end, w_start:w_end]

        if pos_prob is None or random.random() > pos_prob:
            break
        elif mask_patch.sum() > 0:
            break

    return img_patch, mask_patch


def augumentation(input, target):
    if random.random() < 0.5:
        input = input[::-1, :, :]
        target = target[::-1, :]
    if random.random() < 0.5:
        input = input[:, ::-1, :]
        target = target[:, ::-1]
    if random.random() < 0.5:
        input = input.transpose(1, 0, 2)
        target = target.transpose(1, 0)
    return input, target


def PadImg(img, times=32):
    h, w, c = img.shape
    if not h % times == 0:
        img = np.pad(img, ((0, (h // times + 1) * times - h), (0, 0), (0, 0)), mode="constant")
    if not w % times == 0:
        img = np.pad(img, ((0, 0), (0, (w // times + 1) * times - w), (0, 0)), mode="constant")
    return img


def PadMask(img, times=32):
    h, w = img.shape
    if not h % times == 0:
        img = np.pad(img, ((0, (h // times + 1) * times - h), (0, 0)), mode="constant")
    if not w % times == 0:
        img = np.pad(img, ((0, 0), (0, (w // times + 1) * times - w)), mode="constant")
    return img


def to_float_div_255(x):
    return x.float() / 255.0


def directional_sensitive_edge_detection(im_np, mask_np, threshold=33.0, mode="combined"):
    """
    Hybrid spot detection / clutter label generation.
    Supports two modes:
        - "canny": classic Canny with adaptive thresholds + dilated gt exclusion。
        - "combined": Laplacian + TopHat hybrid detection (original behavior).

    Args:
        im_np: [H, W, C] or [H, W] numpy array, uint8/float32
        mask_np: [H, W] numpy array, uint8 (0 or 255/1), Ground Truth Mask
        threshold: Fallback threshold if no target exists
        mode: "canny" or "combined"

    Returns:
        edge_mask: [H, W] numpy array, uint8 (0 or 255)
    """
    # Ensure grayscale
    if len(im_np.shape) == 3:
        img_gray = cv2.cvtColor(im_np, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = im_np

    # Cast for Canny (0-255 uint8)
    img_gray_u8 = img_gray
    if img_gray_u8.dtype != np.uint8:
        img_gray_u8 = np.clip(img_gray_u8, 0, 255).astype(np.uint8)

    if mode == "canny":
        # Adapted from the commented logic in request
        # imgt = im_unnorm * (gt_np > 0)[:, :, None]
        # t1 = im_unnorm.mean()
        # mean_target = imgt[imgt > 0].mean() if imgt.sum() > 0 else 0
        # t2 = abs(mean_target - t1)
        # edge = cv2.Canny(im_unnorm, int(t1), int(t2))
        # blurred = cv2.GaussianBlur(edge, (3, 3), 0)
        # ...

        if len(im_np.shape) == 3:
            imgt = im_np * (mask_np > 0)[:, :, None]
        else:
            imgt = im_np * (mask_np > 0)

        t1 = float(img_gray_u8.mean())
        if imgt.size > 0 and np.count_nonzero(imgt) > 0:
            mean_target = float(imgt[imgt > 0].mean())
        else:
            mean_target = 0.0
        t2 = abs(mean_target - t1)

        low_thr = max(1, min(int(t1), int(t2)))
        high_thr = max(low_thr + 1, max(int(t1), int(t2)))

        edge = cv2.Canny(img_gray_u8, low_thr, high_thr)
        blurred = cv2.GaussianBlur(edge, (3, 3), 0)

        kernel = np.ones((7, 7), np.uint8)
        dilated_gt = cv2.dilate((mask_np > 0).astype(np.uint8), kernel, iterations=1)
        clutter_label_np = ((blurred) > 0).astype(np.uint8) * (1 - dilated_gt)
        return (clutter_label_np.astype(np.uint8) * 255)

    elif mode == "combined":
        if mask_np.sum() > 0:
            kernel_dilate = np.ones((3, 3), np.uint8)
            mask_dilated = cv2.dilate((mask_np > 0).astype(np.uint8), kernel_dilate, iterations=1)
            target_indices = (mask_dilated > 0)
        else:
            target_indices = None

        img_tensor = torch.from_numpy(img_gray).float().unsqueeze(0).unsqueeze(0)
        k_laplace = torch.tensor([[-1, -1, -1],
                                  [-1,  8, -1],
                                  [-1, -1, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        response = torch.nn.functional.conv2d(img_tensor, k_laplace, padding=1)
        response_abs = torch.abs(response).squeeze().numpy()

        if target_indices is not None:
            target_response = response_abs[target_indices]
            lap_threshold = target_response.mean() * 0.5
        else:
            lap_threshold = img_gray.mean() * 0.5

        laplace_mask = (response_abs > lap_threshold)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        tophat = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, kernel)

        if target_indices is not None:
            target_tophat = tophat[target_indices]
            tophat_threshold = target_tophat.mean() * 0.5
        else:
            tophat_threshold = threshold

        tophat_mask = (tophat > tophat_threshold)
        combined_mask = np.logical_or(laplace_mask, tophat_mask).astype(np.uint8) * 255
        combined_mask = cv2.medianBlur(combined_mask, 3)
        return combined_mask

    else:
        raise ValueError(f"directional_sensitive_edge_detection mode must be 'canny' or 'combined', got {mode}")


# Modify from https://github.com/xdFai/SCTransNet/blob/main/dataset.py and https://github.com/YeRen123455/Infrared-Small-Target-Detection
class ImageFolder(Dataset):
    def __init__(
        self,
        path,
        data_set="NUDT",
        istraining=True,
        base_size=256,
        crop_size=256,
        copy_paste=True,
        clutter_mode="combined",
    ):
        self.path = path
        self.copy_paste = copy_paste
        self.clutter_mode = clutter_mode
        self.T_masks = os.path.join(path, data_set, "masks")
        self.T_images = os.path.join(path, data_set, "images")
        self.base_size = base_size
        self.crop_size = crop_size
        self.istraining = istraining
        self.data_set = data_set
        self.images, self.masks = [], []
        self.augumentation = augumentation
        configs_path = "train.txt" if istraining else "test.txt"
        self.totenser = transforms.ToTensor()
        self.train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(to_float_div_255),
            ]
        )

        if data_set == "NUAA-SIRST":
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        elif data_set == "IRSTD-1k":
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.28450727, 0.28450724, 0.28450724], [0.22880708, 0.22880709, 0.22880709]),
                ]
            )
        elif data_set == "NUDT-sea":
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.1583, 0.1583, 0.1583], [0.0885, 0.0885, 0.0885]),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

        # get train set list
        with open(
            os.path.join("./dataset/set_configs", data_set, configs_path),
            encoding="utf-8",
        ) as file:
            lines = file.readlines()

        for line in lines:
            data = line.strip()
            if data_set == "SIRST-UAVB":
                image_path = os.path.join(path, data_set, "images", f"{data}.jpg")
            else:
                image_path = os.path.join(path, data_set, "images", f"{data}.png")
            if data_set == "NUAA-SIRST":
                mask_path = os.path.join(path, data_set, "masks", f"{data}_pixels0.png")
            elif data_set == "SIRST-UAVB":
                mask_path = os.path.join(path, data_set, "masks", f"{data}.jpg")
            else:
                mask_path = os.path.join(path, data_set, "masks", f"{data}.png")
            
            self.images.append(image_path)
            self.masks.append(mask_path)

    def __len__(self):
        return len(self.images)

    def _testval_sync_transform(self, img, mask, clutter_label=None):
        base_size = self.base_size
        if self.data_set == "SIRST-UAVB" or self.data_set == "IRSTDID-SKY":
            img = img.resize((480, 480), Image.BILINEAR)
            mask = mask.resize((480, 480), Image.NEAREST)
            if clutter_label is not None:
                clutter_label = clutter_label.resize((480, 480), Image.NEAREST)
        else:
            img = img.resize((base_size, base_size), Image.BILINEAR)
            mask = mask.resize((base_size, base_size), Image.NEAREST)
            if clutter_label is not None:
                clutter_label = clutter_label.resize((base_size, base_size), Image.NEAREST)

        # final transform
        img, mask = np.array(img), np.array(mask, dtype=np.float32)  # img: <class 'mxnet.ndarray.ndarray.NDArray'> (512, 512, 3)
        if clutter_label is not None:
            clutter_label = np.array(clutter_label, dtype=np.float32)
            return img, mask, clutter_label
        return img, mask

    def _copy_paste_transform(self, img, mask, CP_num):
        img_path = self.T_images + "/"  # img_id的数值正好补了self._image_path在上面定义的2个空
        label_path = self.T_masks + "/"
        w, h = mask.size
        img_dir = os.listdir(img_path)
        label_dir = os.listdir(label_path)
        range_k = len(img_dir)
        dice = random.randint(0, 1)

        if dice == 0:
            img = img
            mask = mask
        else:
            for i in range(CP_num):
                k = random.randint(0, range_k - 1)
                x = random.randint(0, w - 1)
                y = random.randint(0, h - 1)
                T_I_path = img_path + img_dir[k]
                T_M_path = label_path + label_dir[k]
                T_img = Image.open(T_I_path).convert("RGB")  ##由于输入的三通道、单通道图像都有，所以统一转成RGB的三通道，这也符合Unet等网络的期待尺寸
                T_mask = Image.open(T_M_path)
                img.paste(T_img, (x, y))
                mask.paste(T_mask, (x, y))

        return img, mask

    def _sync_transform(self, img, mask, clutter_label=None, is_copy_paste=True):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            if clutter_label is not None:
                clutter_label = clutter_label.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge)
        long_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        if clutter_label is not None:
            clutter_label = clutter_label.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            # fill_color = tuple(np.array(img).mean(axis=(0, 1)).astype(int))
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
            if clutter_label is not None:
                clutter_label = ImageOps.expand(clutter_label, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        if clutter_label is not None:
            clutter_label = clutter_label.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
        # final transform
        if is_copy_paste:
            img, mask = self._copy_paste_transform(img, mask, random.randint(1, 100))  # CP
            # clutter_label logic for copy-paste is complex and usually not done directly on clutter map in same way
            # Assuming we don't copy-paste clutter, or if we do, we need to pass it.
            # For now, let's keep clutter label sync with geometric transforms only.
            
        img, mask = np.array(img), np.array(mask, dtype=np.float32)
        if clutter_label is not None:
            clutter_label = np.array(clutter_label, dtype=np.float32)
            return img, mask, clutter_label
        return img, mask

    def __getitem__(self, index):
        image_path = self.images[index]
        mask_path = self.masks[index]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)
        
        # Ensure image and mask have the same size
        if image.size != mask.size:
            mask = mask.resize(image.size, Image.NEAREST)
            
        mask_array = np.array(mask)
        mask_array[mask_array >= 127] = 255
        mask_array[mask_array < 127] = 0

        mask = Image.fromarray(mask_array)
        image_name = os.path.basename(image_path)

        if self.data_set == "NUDT-sea" or self.data_set == "IRSTD-1k":
            # Calculate clutter label BEFORE sync_transform because we want to augment it too
            # Or calculate it AFTER?
            # User wants _sync_transform to handle clutter_labels.
            # So we must compute clutter_label on the original PIL image first?
            # BUT user's formula uses Canny which operates on image content.
            # The previous implementation calculated it AFTER augmentation on the final tensor/numpy array.
            # If we want to augment clutter_label, we must have it BEFORE augmentation.
            
            # Let's calculate clutter_label on the raw image/mask first, then augment everything together.
            
            # 1. Prepare raw numpy for calculation
            im_np = np.array(image) # RGB
            gt_np = np.array(mask) # 0-255 or 0-1? mask is PIL image mode 'L' or similar. 
            # In __getitem__, mask is already loaded.
            
            # Check mask value range
            # mask_array[mask_array >= 127] = 255
            # mask_array[mask_array < 127] = 0
            # mask is already binary-like 0/255
            gt_np = np.array(mask)
            
            imgt = im_np * (gt_np > 0)[:, :, None]
            kernel = np.ones((5, 5), np.uint8)
            dilated_gt = cv2.dilate((gt_np > 0).astype(np.uint8), kernel, iterations=1)
            target_bg = im_np[(dilated_gt-gt_np) > 0]
            target_pixels = imgt[imgt > 0]
            if len(target_pixels) > 0:
                mean_target = target_pixels.mean()
            else:
                mean_target = 0
            t1 = abs(mean_target - target_bg.mean())
            t2 = abs(mean_target - im_np.mean())
            # edge = cv2.Canny(im_np, int(t1), int(t2)) # Canny expects int thresholds usually
            edge = directional_sensitive_edge_detection(im_np, gt_np, threshold=min(t1, t2), mode=self.clutter_mode)
            blurred = cv2.GaussianBlur(edge, (3, 3), 0)
            clutter_label_np = ((blurred) > 0).astype(np.uint8) * 255 
            
            # Mask out Clutter using dilated GT
            clutter_label_np = clutter_label_np * (1 - dilated_gt)
            clutter_label = Image.fromarray(clutter_label_np.astype(np.uint8))

            if self.istraining:
                img, mask, clutter_label = self._sync_transform(image, mask, clutter_label=clutter_label, is_copy_paste=self.copy_paste)
                img_pil = transforms.ToPILImage()(img.astype(np.uint8)) if isinstance(img, np.ndarray) else img
                img = self.transform(img_pil)
                mask = np.expand_dims(mask, axis=0).astype("float32") / 255.0
                clutter_label = np.expand_dims(clutter_label, axis=0).astype("float32") / 255.0
            else:
                img, mask, clutter_label = self._testval_sync_transform(image, mask, clutter_label=clutter_label)
                img_pil = transforms.ToPILImage()(img.astype(np.uint8)) if isinstance(img, np.ndarray) else img
                img = self.transform(img_pil)
                mask = np.expand_dims(mask, axis=0).astype("float32") / 255.0
                clutter_label = np.expand_dims(clutter_label, axis=0).astype("float32") / 255.0
            
            mask = torch.from_numpy(mask)
            clutter_label = torch.from_numpy(clutter_label)

            return img, mask, clutter_label, image_name
        elif self.data_set == "NUDT-SIRST":
            image = image.resize((self.base_size, self.base_size), Image.BILINEAR)
            mask = mask.resize((self.base_size, self.base_size), Image.BILINEAR)
            if self.istraining:
                image = Normalized(np.array(image, dtype=np.float32), self.data_set)
                mask = np.array(mask, dtype=np.float32) / 255.0
                if len(mask.shape) > 2:
                    mask = mask[:, :, 0]

                img_patch, mask_patch = random_crop(image, mask, self.base_size, pos_prob=0.5)
                img_patch, mask_patch = self.augumentation(img_patch, mask_patch)

                img_patch, mask_patch = (
                    img_patch.transpose(2, 0, 1),
                    mask_patch[np.newaxis, :],
                )
                img = torch.from_numpy(np.ascontiguousarray(img_patch))
                mask = torch.from_numpy(np.ascontiguousarray(mask_patch))
                
                # NUDT-SIRST clutter label calculation (post-augmentation because custom augmentation is numpy-based)
                # Recalculate on the augmented patch
                im_np = (img.numpy().transpose(1, 2, 0)).astype(np.float32) # Normalized image
                # To use Canny, we need uint8 image. We need to un-normalize it to get roughly 0-255 range or just scale.
                # Normalized function: (img - mean) / std. Reverse: img * std + mean.
                # NUDT-SIRST: mean=107.809, std=33.022
                im_unnorm = im_np * 33.02274703979492 + 107.80905151367188
                im_unnorm = np.clip(im_unnorm, 0, 255).astype(np.uint8)
                
                gt_np = (mask.numpy().squeeze() * 255).astype(np.uint8)
                
                # imgt = im_unnorm * (gt_np > 0)[:, :, None]
                # t1 = im_unnorm.mean()
                # mean_target = imgt[imgt > 0].mean() if imgt.sum() > 0 else 0
                # t2 = abs(mean_target - t1)
                #
                # edge = cv2.Canny(im_unnorm, int(t1), int(t2))
                # blurred = cv2.GaussianBlur(edge, (3, 3), 0)
                #
                # # Dilate GT to create a buffer zone
                # kernel = np.ones((7, 7), np.uint8)
                # dilated_gt = cv2.dilate((gt_np > 0).astype(np.uint8), kernel, iterations=1)
                #
                # clutter_label_np = ((blurred) > 0).astype(np.float32) * (1 - dilated_gt)
                # clutter_label = torch.from_numpy(clutter_label_np).unsqueeze(0)

                imgt = im_np * (gt_np > 0)[:, :, None]
                kernel = np.ones((5, 5), np.uint8)
                dilated_gt = cv2.dilate((gt_np > 0).astype(np.uint8), kernel, iterations=1)
                target_bg = im_np[(dilated_gt - gt_np) > 0]
                target_pixels = imgt[imgt > 0]
                if len(target_pixels) > 0:
                    mean_target = target_pixels.mean()
                else:
                    mean_target = 0
                t1 = abs(mean_target - target_bg.mean())
                t2 = abs(mean_target - im_np.mean())
                # edge = cv2.Canny(im_np, int(t1), int(t2)) # Canny expects int thresholds usually
                edge = directional_sensitive_edge_detection(im_np, gt_np, threshold=min(t1, t2), mode=self.clutter_mode)
                blurred = cv2.GaussianBlur(edge, (3, 3), 0)
                clutter_label_np = ((blurred) > 0).astype(np.uint8) * 255

                # Mask out Clutter using dilated GT
                clutter_label_np = clutter_label_np * (1 - dilated_gt)
                clutter_label = Image.fromarray(clutter_label_np.astype(np.uint8))

            else:
                image = Normalized(np.array(image, dtype=np.float32), self.data_set)
                mask = np.array(mask, dtype=np.float32) / 255.0
                if len(mask.shape) > 2:
                    mask = mask[:, :, 0]
                img_patch = PadImg(image)
                mask_patch = PadMask(mask)

                img_patch, mask_patch = (
                    img_patch.transpose(2, 0, 1),
                    mask_patch[np.newaxis, :],
                )
                img = torch.from_numpy(np.ascontiguousarray(img_patch))
                mask = torch.from_numpy(np.ascontiguousarray(mask_patch))
                
                # Recalculate clutter on padded image
                im_np = (img.numpy().transpose(1, 2, 0)).astype(np.float32)
                im_unnorm = im_np * 33.02274703979492 + 107.80905151367188
                im_unnorm = np.clip(im_unnorm, 0, 255).astype(np.uint8)
                
                gt_np = (mask.numpy().squeeze() * 255).astype(np.uint8)
                
                # imgt = im_unnorm * (gt_np > 0)[:, :, None]
                # t1 = im_unnorm.mean()
                # mean_target = imgt[imgt > 0].mean() if imgt.sum() > 0 else 0
                # t2 = abs(mean_target - t1)
                #
                # edge = cv2.Canny(im_unnorm, int(t1), int(t2))
                # blurred = cv2.GaussianBlur(edge, (3, 3), 0)
                #
                # # Dilate GT to create a buffer zone
                # kernel = np.ones((5, 5), np.uint8)
                # dilated_gt = cv2.dilate((gt_np > 0).astype(np.uint8), kernel, iterations=1)
                #
                # clutter_label_np = ((blurred) > 0).astype(np.float32) * (1 - dilated_gt)
                # clutter_label = torch.from_numpy(clutter_label_np).unsqueeze(0)
                imgt = im_np * (gt_np > 0)[:, :, None]
                kernel = np.ones((5, 5), np.uint8)
                dilated_gt = cv2.dilate((gt_np > 0).astype(np.uint8), kernel, iterations=1)
                target_bg = im_np[(dilated_gt - gt_np) > 0]
                target_pixels = imgt[imgt > 0]
                if len(target_pixels) > 0:
                    mean_target = target_pixels.mean()
                else:
                    mean_target = 0
                t1 = abs(mean_target - target_bg.mean())
                t2 = abs(mean_target - im_np.mean())
                # edge = cv2.Canny(im_np, int(t1), int(t2)) # Canny expects int thresholds usually
                edge = directional_sensitive_edge_detection(im_np, gt_np, threshold=min(t1, t2), mode=self.clutter_mode)
                blurred = cv2.GaussianBlur(edge, (3, 3), 0)
                clutter_label_np = ((blurred) > 0).astype(np.uint8) * 255

                # Mask out Clutter using dilated GT
                clutter_label_np = clutter_label_np * (1 - dilated_gt)
                clutter_label = Image.fromarray(clutter_label_np.astype(np.uint8))

            # unify clutter_label output for NUDT-SIRST as tensor [1, H, W]
            if not torch.is_tensor(clutter_label):
                if isinstance(clutter_label, Image.Image):
                    clutter_label = np.array(clutter_label, dtype=np.float32)
                clutter_label = torch.from_numpy((clutter_label / 255.0).astype(np.float32)).unsqueeze(0)

            mask = (mask > 0).to(torch.float32)
            
            return img, mask, clutter_label, image_name
        else:
            # Calculate clutter label FIRST
            im_np = np.array(image)
            gt_np = np.array(mask)
            
            # Check and fix size mismatch
            # if im_np.shape[:2] != gt_np.shape[:2]:
            #     gt_np = cv2.resize(gt_np, (im_np.shape[1], im_np.shape[0]), interpolation=cv2.INTER_NEAREST)
            #     mask = Image.fromarray(gt_np)
            #
            # imgt = im_np * (gt_np > 0)[:, :, None]
            #
            # t1 = im_np.mean()
            # target_pixels = imgt[imgt > 0]
            # if len(target_pixels) > 0:
            #     mean_target = target_pixels.mean()
            # else:
            #     mean_target = 0 # Fallback if no target pixels
            #
            # t2 = abs(mean_target - t1)
            #
            # edge = cv2.Canny(im_np, int(t1), int(t2))
            # blurred = cv2.GaussianBlur(edge, (3, 3), 0)
            #
            # clutter_label_np = ((blurred) > 0).astype(np.uint8) * 255
            #
            # # Dilate GT to create a buffer zone
            # kernel = np.ones((5, 5), np.uint8)
            # dilated_gt = cv2.dilate((gt_np > 0).astype(np.uint8), kernel, iterations=1)
            #
            # # Mask out Clutter using dilated GT
            # clutter_label_np = clutter_label_np * (1 - dilated_gt)
            # clutter_label = Image.fromarray(clutter_label_np.astype(np.uint8))
            imgt = im_np * (gt_np > 0)[:, :, None]
            kernel = np.ones((5, 5), np.uint8)
            dilated_gt = cv2.dilate((gt_np > 0).astype(np.uint8), kernel, iterations=1)
            target_bg = im_np[(dilated_gt - gt_np) > 0]
            target_pixels = imgt[imgt > 0]
            if len(target_pixels) > 0:
                mean_target = target_pixels.mean()
            else:
                mean_target = 0
            t1 = abs(mean_target - target_bg.mean())
            t2 = abs(mean_target - im_np.mean())
            # edge = cv2.Canny(im_np, int(t1), int(t2)) # Canny expects int thresholds usually
            edge = directional_sensitive_edge_detection(im_np, gt_np, threshold=min(t1, t2), mode=self.clutter_mode)
            blurred = cv2.GaussianBlur(edge, (3, 3), 0)
            clutter_label_np = ((blurred) > 0).astype(np.uint8) * 255

            # Mask out Clutter using dilated GT
            clutter_label_np = clutter_label_np * (1 - dilated_gt)
            clutter_label = Image.fromarray(clutter_label_np.astype(np.uint8))

            if self.istraining:
                img, mask, clutter_label = self._sync_transform(image, mask, clutter_label=clutter_label, is_copy_paste=False)
                img_pil = transforms.ToPILImage()(img.astype(np.uint8)) if isinstance(img, np.ndarray) else img
                img = self.transform(img_pil)
                mask = np.expand_dims(mask, axis=0).astype("float32") / 255.0
                clutter_label = np.expand_dims(clutter_label, axis=0).astype("float32") / 255.0
            else:
                img, mask, clutter_label = self._testval_sync_transform(image, mask, clutter_label=clutter_label)
                img_pil = transforms.ToPILImage()(img.astype(np.uint8)) if isinstance(img, np.ndarray) else img
                img = self.transform(img_pil)
                mask = np.expand_dims(mask, axis=0).astype("float32") / 255.0
                clutter_label = np.expand_dims(clutter_label, axis=0).astype("float32") / 255.0

            mask[mask < 1] = 0
            mask = torch.from_numpy(mask)
            clutter_label = torch.from_numpy(clutter_label)

            return img, mask, clutter_label, image_name
