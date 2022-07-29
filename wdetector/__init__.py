import os
import torch
import torch.nn as nn
from torchvision import models

from .fp16module import FP16Module
from .predictor import predict_image, WatermarksPredictor

from huggingface_hub import hf_hub_url, cached_download


MODELS = {
    'resnext101_32x8d-large': dict(
        resnet=models.resnext101_32x8d,
        repo_id='boomb0om/dataset-filters',
        filename='watermark_classifier-resnext101_32x8d-input_size320-4epochs_c097_w082.pth',
    ),
    'resnext50_32x4d-small': dict(
        resnet=models.resnext50_32x4d,
        repo_id='boomb0om/dataset-filters',
        filename='watermark_classifier-resnext50_32x4d-input_size320-4epochs_c082_w078.pth',
    )
}

def get_watermarks_detection_model(name, device='cuda:0', fp16=True, cache_dir='/tmp/watermark-detection'):
    assert name in MODELS
    config = MODELS[name]
    model_ft = config['resnet'](pretrained=False)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)
    
    config_file_url = hf_hub_url(repo_id=config['repo_id'], filename=config['filename'])
    cached_download(config_file_url, cache_dir=cache_dir, force_filename=config['filename'])
    weights = torch.load(os.path.join(cache_dir, config['filename']), device)
    model_ft.load_state_dict(weights)
    
    if fp16:
        model_ft = FP16Module(model_ft)
        
    model_ft.eval()
    model_ft = model_ft.to(device)
    
    return model_ft