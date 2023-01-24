import os
from PIL import Image

IMAGE_EXT = set(['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG'])

def get_extenstion(filepath):
    return os.path.splitext(filepath)[-1]

def listdir_rec(folder_path):
    filepaths = []
    for root, dirname, files in os.walk(folder_path):
        for file in files:
            filepaths.append(os.path.join(root, file))
    return filepaths

def list_images(folder_path):
    files = listdir_rec(folder_path)
    return [f for f in files if get_extenstion(f) in IMAGE_EXT]

def read_image_rgb(path):
    pil_img = Image.open(path)
    pil_img.load()
    if pil_img.format is 'PNG' and pil_img.mode is not 'RGBA':
        pil_img = pil_img.convert('RGBA')
    pil_img = pil_img.convert('RGB')
    return pil_img