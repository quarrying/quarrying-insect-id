import os

import khandy
import numpy as np

from .base import OnnxModel
from .base import check_image_dtype_and_shape


class InsectDetector(OnnxModel):
    def __init__(self, input_width=640, input_height=640):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'models/quarrying_insect_detector.onnx')
        self.input_width = input_width
        self.input_height = input_height
        super(InsectDetector, self).__init__(model_path)

    def _preprocess(self, image):
        check_image_dtype_and_shape(image)
        
         # image size normalization
        image, lb_detail = khandy.letterbox_image(image, self.input_width, self.input_height, 0)
        # image channel normalization
        image = khandy.normalize_image_channel(image, swap_rb=True)
        # image dtype normalization
        image = khandy.rescale_image(image, 'auto', np.float32)
        # to tensor
        image = np.transpose(image, (2,0,1))
        image = np.expand_dims(image, axis=0)
        return image, lb_detail
        
    def _post_process(self, outputs_list, lb_detail, conf_thresh, iou_thresh):
        preds = outputs_list[0][0]
        preds = preds[preds[:, 4] > conf_thresh]
        
        boxes = khandy.convert_boxes_format(preds[:, :4], 'cxcywh', 'xyxy')
        boxes = khandy.unletterbox_2d_points(boxes, lb_detail, False)
        confs = np.max(preds[:, 5:] * preds[:, 4:5], axis=-1)
        classes = np.argmax(preds[:, 5:] * preds[:, 4:5], axis=-1)
        keep = khandy.non_max_suppression(boxes, confs, iou_thresh)
        return boxes[keep], confs[keep], classes[keep]

    def detect(self, image, conf_thresh=0.5, iou_thresh=0.5):
        image, lb_detail = self._preprocess(image)
        outputs_list = self.forward(image)
        boxes, confs, classes = self._post_process(
            outputs_list, 
            lb_detail=lb_detail,
            conf_thresh=conf_thresh, 
            iou_thresh=iou_thresh)
        return boxes, confs, classes
        