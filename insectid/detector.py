import os

import khandy
import numpy as np

from .base import OnnxModel


class InsectDetector(OnnxModel):
    def __init__(self, input_width=640, input_height=640):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'models/quarrying_insect_detector.onnx')
        self.input_width = input_width
        self.input_height = input_height
        super(InsectDetector, self).__init__(model_path)

    def _preprocess(self, image):
        image_dtype = image.dtype
        assert image_dtype in [np.uint8, np.uint16]

        image = khandy.normalize_image_shape(image, swap_rb=True)
        image, scale, pad_left, pad_top = khandy.letterbox_image(
            image, self.input_width, self.input_height, 0, return_scale=True)
        image = image.astype(np.float32)
        if image_dtype == np.uint8:
            image /= 255.0
        else:
            image /= 65535.0
        image = np.transpose(image, (2,0,1))
        image = np.expand_dims(image, axis=0)
        return image, scale, pad_left, pad_top
        
    def _post_process(self, outputs_list, scale, pad_left, pad_top, conf_thresh, iou_thresh):
        pred = outputs_list[0][0]
        pass_t = pred[:, 4] > conf_thresh
        pred = pred[pass_t]
        
        boxes = khandy.convert_boxes_format(pred[:, :4], 'cxcywh', 'xyxy')
        boxes = khandy.unletterbox_2d_points(boxes, scale, pad_left, pad_top, False)
        confs = np.max(pred[:, 5:] * pred[:, 4:5], axis=-1)
        classes = np.argmax(pred[:, 5:] * pred[:, 4:5], axis=-1)
        keep = khandy.non_max_suppression(boxes, confs, iou_thresh)
        return boxes[keep], confs[keep], classes[keep]

    def detect(self, image, conf_thresh=0.5, iou_thresh=0.5):
        image, scale, pad_left, pad_top = self._preprocess(image)
        outputs_list = self.forward(image)
        boxes, confs, classes = self._post_process(
            outputs_list, 
            scale=scale, 
            pad_left=pad_left,
            pad_top=pad_top,
            conf_thresh=conf_thresh, 
            iou_thresh=iou_thresh)
        return boxes, confs, classes
        