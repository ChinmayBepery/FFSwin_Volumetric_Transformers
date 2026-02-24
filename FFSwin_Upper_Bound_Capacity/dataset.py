import os, glob, cv2, torch
import numpy as np
from torch.utils.data import Dataset

class BalancedOCTVolumeDataset(Dataset):
    def __init__(self, root_dir, img_size=256, return_label=False):
        self.img_size = img_size
        self.return_label = return_label
        self.folders = []
        
        # 1. Get folders by class
        dry = self._get_subs(os.path.join(root_dir, 'DryAMD'))
        wet = self._get_subs(os.path.join(root_dir, 'WetAMD'))
        non = self._get_subs(os.path.join(root_dir, 'NonAMD'))
        
        # 2. Balance (Oversample to match the majority)
        target = max(len(dry), len(wet), len(non))
        if target == 0: target = 1
        
        self.folders_dry = self._oversample(dry, target)
        self.folders_wet = self._oversample(wet, target)
        self.folders_non = self._oversample(non, target)
        
        # 3. Combine into one list
        self.folders = self.folders_dry + self.folders_wet + self.folders_non
        
        # 4. Create corresponding labels if needed (0=Dry, 1=Wet, 2=NonAMD)
        if self.return_label:
            self.labels = [0]*len(self.folders_dry) + [1]*len(self.folders_wet) + [2]*len(self.folders_non)

    def _get_subs(self, p): return [f.path for f in os.scandir(p) if f.is_dir()] if os.path.exists(p) else []
    
    def _oversample(self, lst, n):
        if not lst: return []
        out = []
        while len(out) < n: out.extend(lst)
        return out[:n]

    def __len__(self): return len(self.folders)
    
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
            if img is not None: vol.append(cv2.resize(img, (self.img_size, self.img_size)))
        
        if not vol: return torch.zeros(1, 1, 16, self.img_size, self.img_size)
        
        # Pad 33 -> 34 (For Swin)
        if len(vol) % 2 != 0: vol.append(vol[-1])
            
        return torch.FloatTensor(np.array(vol)/255.0).unsqueeze(0)

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

    def __len__(self): return len(self.folders)
    
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
            if img is not None: vol.append(cv2.resize(img, (self.img_size, self.img_size)))
        
        if not vol: return torch.zeros(1, 1, 16, self.img_size, self.img_size)
        if len(vol) % 2 != 0: vol.append(vol[-1]) 
        return torch.FloatTensor(np.array(vol)/255.0).unsqueeze(0)
