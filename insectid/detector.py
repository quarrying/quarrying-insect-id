import os

import cv2
import khandy
import numpy as np

from .base import OnnxModel
from .base import normalize_image_shape


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

        image = normalize_image_shape(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image, scale, left, top = khandy.letterbox_resize_image(image, 
                                                                self.input_width, 
                                                                self.input_height, 
                                                                0,
                                                                return_scale=True)
        image = image.astype(np.float32)
        if image_dtype == np.uint8:
            image /= 255.0
        else:
            image /= 65535.0
        image = np.transpose(image, (2,0,1))
        image = np.expand_dims(image, axis=0)
        return image, scale, left, top
        
    def _post_process(self, outputs_list, scale, left, top, conf_thresh, iou_thresh):
        pred = outputs_list[0][0]
        pass_t = pred[..., 4] > conf_thresh
        pred = pred[pass_t]

        boxes = self._cxcywh2xyxy(pred[..., :4], scale, left, top)
        confs = np.amax(pred[:, 5:] * pred[:, 4:5], axis=-1)
        classes = np.argmax(pred[:, 5:] * pred[:, 4:5], axis=-1)
        keep = khandy.non_max_suppression(boxes, confs, iou_thresh)
        return boxes[keep], confs[keep], classes[keep]

    def _cxcywh2xyxy(self, x, scale, left, top):
        y = np.zeros_like(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2 - left
        y[:, 2] = x[:, 0] + x[:, 2] / 2 - left
        y[:, 1] = x[:, 1] - x[:, 3] / 2 - top
        y[:, 3] = x[:, 1] + x[:, 3] / 2 - top
        y /= scale
        return y
        
    def detect(self, image, conf_thresh=0.5, iou_thresh=0.5):
        resized, scale, left, top = self._preprocess(image)
        outputs_list = self.forward(resized)
        boxes, confs, classes = self._post_process(outputs_list, 
                                                   scale=scale, 
                                                   left=left,
                                                   top=top,
                                                   conf_thresh=conf_thresh, 
                                                   iou_thresh=iou_thresh)
        return boxes, confs, classes
        