import os
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import string 
import random

CV2_FONTS = [
    #cv2.FONT_HERSHEY_COMPLEX,
    cv2.FONT_HERSHEY_COMPLEX_SMALL,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_PLAIN,
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_TRIPLEX,
    cv2.FONT_ITALIC,
    cv2.QT_FONT_BLACK,
    cv2.QT_FONT_NORMAL
]

# рандомный float между x и y
def random_float(x, y):
    return random.random()*(y-x)+x

# вычисляет размер текста в пикселях для cv2.putText
def get_text_size(text, font, font_scale, thickness):
    (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    return w, h+baseline

# вычисляет какой нужен font_scale для определенного размера текста (по высоте)
def get_font_scale(needed_height, text, font, thickness):
    w, h = get_text_size(text, font, 1, thickness)
    return needed_height/h

# добавляет текст на изображение
def place_text(image, text, color=(255,255,255), alpha=1, position=(0, 0), angle=0,
               font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1.0, thickness=3):
    image = np.array(image)
    overlay = np.zeros_like(image)
    output = image.copy()
    
    cv2.putText(overlay, text, position, font, font_scale, color, thickness)
    
    if angle != 0:
        text_w, text_h = get_text_size(text, font, font_scale, thickness)
        rotate_M = cv2.getRotationMatrix2D((position[0]+text_w//2, position[1]-text_h//2), angle, 1)
        overlay = cv2.warpAffine(overlay, rotate_M, (overlay.shape[1], overlay.shape[0]))
    
    overlay[overlay==0] = image[overlay==0]
    cv2.addWeighted(overlay, alpha, output, 1-alpha, 0, output)
    
    return Image.fromarray(output)

def get_random_font_params(text, text_height, fonts, font_thickness_range):
    font = random.choice(fonts)
    font_thickness_range_scaled = [int(font_thickness_range[0]*(text_height/35)),
                                   int(font_thickness_range[1]*(text_height/85))]
    try:
        font_thickness = min(random.randint(*font_thickness_range_scaled), 2)
    except ValueError:
        font_thickness = 2
    font_scale = get_font_scale(text_height, text, font, font_thickness)
    return font, font_scale, font_thickness

# устанавливает вотермарку в центре изображения с рандомными параметрами
def place_random_centered_watermark(
        pil_image, 
        text,
        center_point_range_shift=(-0.025, 0.025),
        random_angle=(0,0),
        text_height_in_percent_range=(0.15, 0.18),
        text_alpha_range=(0.23, 0.5),
        fonts=CV2_FONTS,
        font_thickness_range=(2, 7),
        colors=[(255,255,255)]
    ):
    w, h = pil_image.size
    
    position_shift_x = random_float(*center_point_range_shift)
    offset_x = int(w*position_shift_x)
    position_shift_y = random_float(*center_point_range_shift)
    offset_y = int(w*position_shift_y)
    
    text_height = int(h*random_float(*text_height_in_percent_range))
    
    font, font_scale, font_thickness = get_random_font_params(text, text_height, fonts, font_thickness_range)

    text_width, _ = get_text_size(text, font, font_scale, font_thickness)
    
    position_x = int((w/2)-text_width/2+offset_x)
    position_y = int((h/2)+text_height/2+offset_y)
    
    return place_text(
        pil_image, 
        text,
        color=random.choice(colors),
        alpha=random_float(*text_alpha_range),
        position=(position_x, position_y), 
        angle=random.randint(*random_angle),
        thickness=font_thickness,
        font=font, 
        font_scale=font_scale
    )

def place_random_watermark(
        pil_image, 
        text,
        random_angle=(0,0),
        text_height_in_percent_range=(0.10, 0.18),
        text_alpha_range=(0.18, 0.4),
        fonts=CV2_FONTS,
        font_thickness_range=(2, 6),
        colors=[(255,255,255)]
    ):
    w, h = pil_image.size
    
    text_height = int(h*random_float(*text_height_in_percent_range))
    
    font, font_scale, font_thickness = get_random_font_params(text, text_height, fonts, font_thickness_range)

    text_width, _ = get_text_size(text, font, font_scale, font_thickness)
    
    position_x = random.randint(0, max(w-text_width, 10))
    position_y = random.randint(text_height, h)
    
    return place_text(
            pil_image, 
            text,
            color=random.choice(colors),
            alpha=random_float(*text_alpha_range),
            position=(position_x, position_y), 
            angle=random.randint(*random_angle),
            thickness=font_thickness,
            font=font, 
            font_scale=font_scale
        )

def center_crop(image, w, h):
    center = image.shape
    x = center[1]/2 - w/2
    y = center[0]/2 - h/2
    return image[int(y):int(y+h), int(x):int(x+w)]

# добавляет текст в шахматном порядке на изображение
def place_text_checkerboard(image, text, color=(255,255,255), alpha=1, step_x=0.1, step_y=0.1, angle=0,
                            font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1.0, thickness=3):
    image_size = image.size
    
    image = np.array(image.convert('RGB'))
    if angle != 0:
        border_scale = 0.4
        overlay_size = [int(i*(1+border_scale)) for i in list(image_size)]
    else:
        overlay_size = image_size
        
    w, h = overlay_size
    overlay = np.zeros((overlay_size[1], overlay_size[0], 3)) # change dimensions
    output = image.copy()
    
    text_w, text_h = get_text_size(text, font, font_scale, thickness)
    
    c = 0
    for rel_pos_x in np.arange(0, 1, step_x):
        c += 1
        for rel_pos_y in np.arange(text_h/h+(c%2)*step_y/2, 1, step_y):
            position = (int(w*rel_pos_x), int(h*rel_pos_y))
            cv2.putText(overlay, text, position, font, font_scale, color, thickness)
    
    if angle != 0:
        rotate_M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
        overlay = cv2.warpAffine(overlay, rotate_M, (overlay.shape[1], overlay.shape[0]))
    
    overlay = center_crop(overlay, image_size[0], image_size[1])
    overlay[overlay==0] = image[overlay==0]
    overlay = overlay.astype(np.uint8)
    cv2.addWeighted(overlay, alpha, output, 1-alpha, 0, output)
    
    return Image.fromarray(output)

def place_random_diagonal_watermark(
        pil_image, 
        text,
        random_step_x=(0.25, 0.4),
        random_step_y=(0.25, 0.4),
        random_angle=(-60,60),
        text_height_in_percent_range=(0.10, 0.18),
        text_alpha_range=(0.18, 0.4),
        fonts=CV2_FONTS,
        font_thickness_range=(2, 6),
        colors=[(255,255,255)]
    ):
    w, h = pil_image.size
    
    text_height = int(h*random_float(*text_height_in_percent_range))
    
    font, font_scale, font_thickness = get_random_font_params(text, text_height, fonts, font_thickness_range)

    text_width, _ = get_text_size(text, font, font_scale, font_thickness)
    
    return place_text_checkerboard(
            pil_image, 
            text,
            color=random.choice(colors),
            alpha=random_float(*text_alpha_range),
            step_x=random_float(*random_step_x),
            step_y=random_float(*random_step_y),
            angle=random.randint(*random_angle),
            thickness=font_thickness,
            font=font, 
            font_scale=font_scale
        )