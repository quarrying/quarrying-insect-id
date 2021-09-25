import os

import cv2
import khandy
import numpy as np

from .base import OnnxModel


def non_max_suppression(boxes, scores, iou_thresh=0.3):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1) 
    order = scores.flatten().argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        overlap_ratio = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(overlap_ratio <= iou_thresh)[0]
        order = order[inds + 1]
    return keep
    
    
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
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = np.ascontiguousarray(image)
        return image, scale, left, top
        
    def _post_process(self, outputs_list, scale=0, left=0, top=0, 
                      conf_thresh=0.5, iou_thresh=0.5):
        pred = outputs_list[0][0]
        pass_t = pred[..., 4] > conf_thresh
        pred = pred[pass_t]

        boxes = self._cxcywh2xyxy(pred[..., :4], scale, left, top)
        confs = np.amax(pred[:, 5:] * pred[:, 4:5], axis=-1, keepdims=True)
        classes = np.argmax(pred[:, 5:], axis=-1)
        keep = non_max_suppression(boxes, confs, iou_thresh)
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
        