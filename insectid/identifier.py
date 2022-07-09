import os
import copy
from collections import OrderedDict

import khandy
import numpy as np

from .base import OnnxModel


class InsectIdentifier(OnnxModel):
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'models/quarrying_insect_identifier.onnx')
        label_map_path = os.path.join(current_dir, 'models/quarrying_insectid_label_map.txt')
        super(InsectIdentifier, self).__init__(model_path)
        
        self.label_name_dict = self._get_label_name_dict(label_map_path)
        self.names = [self.label_name_dict[i]['chinese_name'] for i in range(len(self.label_name_dict))]
        
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
        image_dtype = image.dtype
        assert image_dtype in [np.uint8, np.uint16]
        
        image = khandy.normalize_image_shape(image, swap_rb=True)
        image = khandy.letterbox_resize_image(image, 224, 224)
        image = image.astype(np.float32)
        if image_dtype == np.uint8:
            image /= 255.0
        else:
            image /= 65535.0
        image -= np.asarray([0.485, 0.456, 0.406])
        image /= np.asarray([0.229, 0.224, 0.225])
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
        if topk <= 0:
            topk = len(self.label_name_dict)
            
        results = []
        probs = self.predict(image)
        taxon_topk = min(probs.shape[-1], topk)
        topk_probs, topk_indices = khandy.top_k(probs, taxon_topk)
        for ind, prob in zip(topk_indices[0], topk_probs[0]):
            one_result = copy.deepcopy(self.label_name_dict[ind])
            one_result['probability'] = prob
            results.append(one_result)   
        return results
        
        