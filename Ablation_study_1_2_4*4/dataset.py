import os
import glob
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class BalancedOCTVolumeDataset(Dataset):
    def __init__(self, root_dir, img_size=256, return_label=False):
        self.img_size = img_size
        self.return_label = return_label
        self.folders = []

        dry = self._get_subs(os.path.join(root_dir, 'DryAMD'))
        wet = self._get_subs(os.path.join(root_dir, 'WetAMD'))
        non = self._get_subs(os.path.join(root_dir, 'NonAMD'))

        target = max(len(dry), len(wet), len(non))
        if target == 0:
            target = 1

        self.folders_dry = self._oversample(dry, target)
        self.folders_wet = self._oversample(wet, target)
        self.folders_non = self._oversample(non, target)

        self.folders = self.folders_dry + self.folders_wet + self.folders_non

        if self.return_label:
            self.labels = [0]*len(self.folders_dry) + [1]*len(self.folders_wet) + [2]*len(self.folders_non)

    def _get_subs(self, path):
        if not os.path.exists(path):
            return []
        return [f.path for f in os.scandir(path) if f.is_dir()]

    def _oversample(self, lst, n):
        if not lst:
            return []
        out = []
        while len(out) < n:
            out.extend(lst)
        return out[:n]

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        vol = self._load(self.folders[idx])
        if self.return_label:
            return vol, self.labels[idx]
        return vol

    def _load(self, path):
        files = sorted(glob.glob(os.path.join(path, "*.bmp")))
        vol = []
        for f in files:
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (self.img_size, self.img_size))
                vol.append(img)

        if len(vol) == 0:
            return torch.zeros(1, 40, self.img_size, self.img_size, dtype=torch.float32)

        # Pad to exactly 40 slices (to work with depth_patch = 1,2,4)
        target_depth = 40
        while len(vol) < target_depth:
            vol.append(vol[-1])
        vol = vol[:target_depth]

        vol = np.array(vol, dtype=np.float32) / 255.0
        return torch.FloatTensor(vol).unsqueeze(0)   # (1, 40, H, W)


class SimpleOCTDataset(Dataset):
    def __init__(self, root_dir, img_size=256, return_label=False):
        self.img_size = img_size
        self.return_label = return_label
        self.folders = []
        self.labels = []

        label_map = {'DryAMD': 0, 'WetAMD': 1, 'NonAMD': 2}

        for cat in ['DryAMD', 'WetAMD', 'NonAMD']:
            p = os.path.join(root_dir, cat)
            if os.path.exists(p):
                subs = [f.path for f in os.scandir(p) if f.is_dir()]
                self.folders.extend(subs)
                self.labels.extend([label_map[cat]] * len(subs))

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        vol = self._load(self.folders[idx])
        if self.return_label:
            return vol, self.labels[idx]
        return vol, self.folders[idx]

    def _load(self, path):
        files = sorted(glob.glob(os.path.join(path, "*.bmp")))
        vol = []
        for f in files:
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (self.img_size, self.img_size))
                vol.append(img)

        if len(vol) == 0:
            return torch.zeros(1, 40, self.img_size, self.img_size, dtype=torch.float32)

        target_depth = 40
        while len(vol) < target_depth:
            vol.append(vol[-1])
        vol = vol[:target_depth]

        vol = np.array(vol, dtype=np.float32) / 255.0
        return torch.FloatTensor(vol).unsqueeze(0)