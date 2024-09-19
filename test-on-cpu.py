from PIL import Image
from wmdetection.models import get_watermarks_detection_model
from wmdetection.pipelines.predictor import WatermarksPredictor
import os

# checkpoint is automatically downloaded
model, transforms = get_watermarks_detection_model(
    "convnext-tiny", device="cpu", fp16=False, cache_dir="./weights"
)
predictor = WatermarksPredictor(model, transforms, "cpu")


for root, dirs, files in os.walk("testfolder"):
    for file in files:
        path_to_image = os.path.join(root, file)
        result = predictor.predict_image(Image.open(path_to_image))
        print(f"{path_to_image}: {'watermarked' if result else 'clean'}")
