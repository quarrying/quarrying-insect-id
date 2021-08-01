import os

import cv2
import khandy
import numpy as np
import PIL
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from insectid import InsectDetector
from insectid import InsectIdentifier


def draw_rectangles(image, boxes):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    for i in range(x1.shape[0]):
        cv2.rectangle(image, (int(x1[i]), int(y1[i])), (int(x2[i]), int(y2[i])), 
                      (0,255,0), 2)
    return image
    
                    
def imread_ex(filename, flags=-1):
    try:
        return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), flags)
    except Exception as e:
        return None
        
        
def draw_text(image, text, position, font_size=15, color=(255,0,0),
              font_filename='data/simsun.ttc'):
    assert isinstance(color, (tuple, list)) and len(color) == 3
    gray = color[0]
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image)
        color = (color[2], color[1], color[0])
    elif isinstance(image, PIL.Image.Image):
        pil_image = image
    else:
        raise ValueError('Unsupported image type!')
    assert pil_image.mode in ['L', 'RGB']
    if pil_image.mode == 'L':
        color = gray

    font_object = ImageFont.truetype(font_filename, size=font_size)
    drawable = ImageDraw.Draw(pil_image)
    drawable.text((position[0], position[1]), text, 
                  fill=color, font=font_object)

    if isinstance(image, np.ndarray):
        return np.asarray(pil_image)
    return pil_image


if __name__ == '__main__':
    src_dir = r'F:\_Data\Nature\_raw\_insect'
    
    detector = InsectDetector()
    identifier = InsectIdentifier()
    src_filenames = khandy.get_all_filenames(src_dir)
    for k, filename in enumerate(src_filenames):
        print('[{}/{}] {}'.format(k+1, len(src_filenames), filename))
        image = imread_ex(filename)
        if image is None:
            continue
        if max(image.shape[:2]) > 1280:
            image = khandy.resize_image_long(image, 1280)
        image_for_draw = image.copy()
        image_height, image_width = image.shape[:2]
        
        boxes, confs, classes = detector.detect(image)
        draw_rectangles(image_for_draw, boxes)
        for box, conf, class_ind in zip(boxes, confs, classes):
            cropped = khandy.crop_or_pad(image, int(box[0]), int(box[1]), int(box[2]), int(box[3]))
            outputs = identifier.identify(cropped)
            print(outputs['results'][0])
            if outputs['status'] == 0:
                prob = outputs['results'][0]['probability']
                if prob < 0.10:
                    text = 'Unknown'
                else:
                    text = '{}: {:.3f}'.format(outputs['results'][0]['chinese_name'], 
                                               outputs['results'][0]['probability'])
                position = [int(box[0] + 2), int(box[1] - 20)]
                position[0] = min(max(position[0], 0), image_width)
                position[1] = min(max(position[1], 0), image_height)
                image_for_draw = draw_text(image_for_draw, text, position)
        cv2.imshow('image', image_for_draw)
        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyAllWindows()
            break
            
        