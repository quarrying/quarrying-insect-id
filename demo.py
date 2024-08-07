import os
import time

import cv2
import khandy
import numpy as np

from insectid import InsectDetector
from insectid import InsectIdentifier


if __name__ == '__main__':
    src_dirs = [r'images', r'E:\_raw\_insect']
    
    detector = InsectDetector()
    identifier = InsectIdentifier()
    src_filenames = sum([khandy.list_files_in_dir(src_dir, True) for src_dir in src_dirs], [])
    src_filenames = sorted(src_filenames, key=lambda t: os.stat(t).st_mtime, reverse=True)
    
    for k, filename in enumerate(src_filenames):
        print('[{}/{}] {}'.format(k+1, len(src_filenames), filename))
        start_time = time.time()
        image = khandy.imread(filename)
        if image is None:
            continue
        if max(image.shape[:2]) > 1280:
            image = khandy.resize_image_long(image, 1280)
        image_for_draw = image.copy()
        image_height, image_width = image.shape[:2]
        
        boxes, confs, classes = detector.detect(image)
        for box, conf, class_ind in zip(boxes, confs, classes):
            box = box.astype(np.int32)
            box_width = box[2] - box[0] 
            box_height = box[3] - box[1]
            if box_width < 30 or box_height < 30:
                continue
                
            cropped = khandy.crop_or_pad(image, box[0], box[1], box[2], box[3])
            results = identifier.identify(cropped)
            print(results[0])
            prob = results[0]['probability']
            if prob < 0.10:
                text = 'Unknown'
            else:
                text = '{}: {:.3f}'.format(results[0]['chinese_name'], results[0]['probability'])
            position = [box[0] + 2, box[1] - 20]
            position[0] = min(max(position[0], 0), image_width)
            position[1] = min(max(position[1], 0), image_height)
            cv2.rectangle(image_for_draw, (box[0], box[1]), (box[2], box[3]), (0,255,0), 2)
            image_for_draw = khandy.draw_text(image_for_draw, text, position, 
                                              font='simsun.ttc', font_size=15)

        print('Elapsed: {:.3f}s'.format(time.time() - start_time))
        cv2.imshow('image', image_for_draw)
        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyAllWindows()
            break
            
        