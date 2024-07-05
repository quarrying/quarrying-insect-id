import os
import copy
from collections import OrderedDict

import khandy
import numpy as np

from .base import OnnxModel
from .base import check_image_dtype_and_shape


class InsectIdentifier(OnnxModel):
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'models/quarrying_insect_identifier.onnx')
        label_map_path = os.path.join(current_dir, 'models/quarrying_insectid_label_map.txt')
        super(InsectIdentifier, self).__init__(model_path)
        
        self.label_name_dict = self._get_label_name_dict(label_map_path)
        self.names = [self.label_name_dict[i]['chinese_name'] for i in range(len(self.label_name_dict))]
        self.num_classes = len(self.label_name_dict)

    @staticmethod
    def _get_label_name_dict(filename):
        records = khandy.load_list(filename)
        label_name_dict = {}
        for record in records:
            label, chinese_name, latin_name = record.split(',')
            label_name_dict[int(label)] = OrderedDict([('chinese_name', chinese_name), 
                                                       ('latin_name', latin_name)])
        return label_name_dict
        
    @staticmethod
    def _preprocess(image):
        check_image_dtype_and_shape(image)
        
        # image size normalization
        image, _ = khandy.letterbox_image(image, 224, 224)
        # image channel normalization
        image = khandy.normalize_image_channel(image, swap_rb=True)
        # image dtype normalization
        # image dtype and value range normalization
        mean, stddev = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        image = khandy.normalize_image_value(image, mean, stddev, 'auto')
        # to tensor
        image = np.transpose(image, (2,0,1))
        image = np.expand_dims(image, axis=0)
        return image
        
    def predict(self, image):
        inputs = self._preprocess(image)
        logits = self.forward(inputs)
        probs = khandy.softmax(logits)
        return probs
        
    def identify(self, image, topk=5):
        assert isinstance(topk, int)
        if topk <= 0 or topk > self.num_classes:
            topk = self.num_classes
            
        probs = self.predict(image)
        topk_probs, topk_indices = khandy.top_k(probs, topk)

        results = []
        for ind, prob in zip(topk_indices[0], topk_probs[0]):
            one_result = copy.deepcopy(self.label_name_dict[ind])
            one_result['probability'] = prob
            results.append(one_result)
        return results
        
        