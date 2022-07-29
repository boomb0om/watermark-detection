import os
from tqdm import tqdm
from PIL import Image
import numpy as np

import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.utils.data import BatchSampler, DataLoader

from .utils import read_image_rgb, vprint


classifier_transforms = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(pil_image, model, device):
    input_img = classifier_transforms(pil_image).float().unsqueeze(0)
    outputs = model(input_img.to(device))
    result = torch.max(outputs, 1)[1].cpu().reshape(-1).tolist()[0]
    return result


class ImageDataset(Dataset):
    
    def __init__(self, objects):
        self.objects = objects
        self.resnet_transforms = classifier_transforms
        
    def __len__(self):
        return len(self.objects)

    def __getitem__(self, idx):
        obj = self.objects[idx]
        assert isinstance(obj, (str, np.ndarray, Image.Image))
        if isinstance(obj, str):
            pil_img = read_image_rgb(obj)
        elif isinstance(obj, np.ndarray):
            pil_img = Image.fromarray(obj)
        elif isinstance(obj, Image.Image):
            pil_img = obj
        
        resnet_img = self.resnet_transforms(pil_img).float()
        
        return resnet_img
    
    
class WatermarksPredictor:
    
    def __init__(self, resnet_model, device, workers=8, bs=8, verbose=False):
        """
        Predict with watermark classifier using batches and torch.DataLoader
        resnet_model: watermark classifier model
        device: torch.device to use
        workers: number of workers for dataloader
        bs: batch size to use
        verbose: print additional info or not
        """
        
        self.resnet_model = resnet_model
        self.num_workers = workers
        self.bs = bs
        self.device = device
        self.verbose = verbose
        
        vprint(f'Using device {self.device}', verbose=self.verbose)
        self.resnet_model = resnet_model
        
    def run(self, files):
        """
        Processes input objects (list of paths, list of PIL images, list of numpy arrays) and returns model results.
        files: objects to process. Should be list of paths to images or list of PIL images or list of numpy arrays
        """
        vprint(f'Files to process: {len(files)}', verbose=self.verbose)
        
        eval_dataset = ImageDataset(files)
        loader = DataLoader(
            eval_dataset,
            sampler=torch.utils.data.SequentialSampler(eval_dataset),
            batch_size=self.bs,
            drop_last=False,
            num_workers=self.num_workers
        )
        if self.verbose:
            loader = tqdm(loader)
        
        result = []
        for batch in loader:
            with torch.no_grad():
                outputs = self.resnet_model(batch.to(self.device))
                result.extend(torch.max(outputs, 1)[1].cpu().reshape(-1).tolist())
        
        return result