# watermark-detection

- [Installation](#installation)
- [Basic usage](#basic-usage)
- Validation dataset
- Training and train dataset
  - training dataset
  - training a model

## Installation

```bash
git clone https://github.com/boomb0om/watermark-detection
cd watermark-detection
pip install -r requirements.txt
```

Model weights are automatically downloaded when you create model with `get_watermarks_detection_model`.
Also you can find them on [huggingface](https://huggingface.co/boomb0om/watermark-detectors)

## Basic usage

Checkout [this notebook](https://github.com/boomb0om/watermark-detection/blob/main/jupyters/run_model.ipynb) for basic usage.

```python
from PIL import Image
from wmdetection.models import get_watermarks_detection_model
from wmdetection.pipelines.predictor import WatermarksPredictor

# checkpoint is automatically downloaded
model, transforms = get_watermarks_detection_model(
    'convnext-tiny', 
    fp16=False, 
    cache_dir='/path/to/weights'
)
predictor = WatermarksPredictor(model, transforms, 'cuda:0')

result = predictor.predict_image(Image.open('images/watermark/1.jpg'))
print('watermarked' if result else 'clean') # prints "watermarked"
```

Use `predictor.run` to run model on list of images:

```python
results = predictor.run([
    'images/watermark/1.jpg',
    'images/watermark/2.jpg',
    'images/watermark/3.jpg',
    'images/watermark/4.jpg',
    'images/clear/1.jpg',
    'images/clear/2.jpg',
    'images/clear/3.jpg',
    'images/clear/4.jpg'
], num_workers=8, bs=8)
for result in results:
    print('watermarked' if result else 'clean')
```

## Validation dataset


## Training and train dataset

### training dataset

### training a model
