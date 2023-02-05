# watermark-detection

- [Installation](#installation)
- [Basic usage](#basic-usage)
- [Test dataset](#test-dataset)
- [Training a model and train dataset](#training-and-train-dataset)
  - [Training dataset](#training-dataset)
  - [Training a model](#training-a-model)

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

## Test dataset

Download test dataset from [huggingface](https://huggingface.co/datasets/boomb0om/watermarks-validation) and put it into `dataset/` folder. Then run evaluation code in [this jupyter](https://github.com/boomb0om/watermark-detection/blob/main/jupyters/evaluate_model.ipynb).

Test dataset is consists of 60 clean images and 62 watermarked images. Model accuracy for this dataset is shown below:

| **Model name** | **Accuracy** |
|---|---|
| convnext-tiny | 93,44% |
| resnext101_32x8d-large | 84,42% |
| [ARKseal model](https://github.com/ARKseal/watermark-detection) | 77,86% |
| resnext50_32x4d-small | 76,22% |

## Training and train dataset

### Training dataset

Synthetic training data is generated using random watermark generator. Gather clean images for generator and put them into `dataset/` folder. Then run [notebook with synthetic dataset generation](https://github.com/boomb0om/watermark-detection/blob/main/jupyters/generate_dataset.ipynb). Generator will randomly put watermark on every image and save it into another folder.

After synthetic images are generated, run [notebook](https://github.com/boomb0om/watermark-detection/blob/main/dataset/create_train_csv.ipynb) to create csv for all training images.

### Training a model

You can find traning code [in this notebook](https://github.com/boomb0om/watermark-detection/blob/main/jupyters/train.ipynb). Good luck!
