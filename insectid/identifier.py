import os
from collections import OrderedDict

import cv2
import khandy
import numpy as np
import onnxruntime as rt


def normalize_image_shape(image):
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3:
        num_channels = image.shape[-1]
        if num_channels == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif num_channels == 3:
            pass
        elif num_channels == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        else:
            raise ValueError('Unsupported!')
    else:
        raise ValueError('Unsupported!')
    return image
    
    
class InsectIdentifier(object):
    def __init__(self):
        curr_dir = os.path.dirname(__file__)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_filename = os.path.join(current_dir, 'models/quarrying_insect_identifier.onnx')
        label_map_filename = os.path.join(current_dir, 'models/quarrying_insectid_label_map.txt')
        
        self.sess = rt.InferenceSession(model_filename)
        self.input_names = [item.name for item in self.sess.get_inputs()]
        self.output_names = [item.name for item in self.sess.get_outputs()]
        self.label_name_dict = self._get_label_name_dict(label_map_filename)
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
        
        image = normalize_image_shape(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = khandy.letterbox_resize_image(image, 299, 299)
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
        try:
            inputs = self._preprocess(image)
        except:
            return {"status": -1, "message": "Inference preprocess error.", "results": {}}
        
        try:
            logits = self.sess.run(self.output_names, {self.input_names[0]: inputs})
            probs = khandy.softmax(logits[0])
        except:
            return {"status": -2, "message": "Inference error.", "results": {}}
        results = {'probs': probs}
        return {"status": 0, "message": "OK", "results": results}
        
    def identify(self, image, topk=5):
        assert isinstance(topk, int)
        if topk <= 0:
            topk = len(self.label_name_dict)
            
        results = []
        outputs = self.predict(image)
        status = outputs['status']
        message = outputs['message']
        if outputs['status'] != 0:
            return {"status": status, "message": message, "results": results}
            
        probs = outputs['results']['probs']
        taxon_topk = min(probs.shape[-1], topk)
        topk_probs, topk_indices = khandy.top_k(probs, taxon_topk)
        for ind, prob in zip(topk_indices[0], topk_probs[0]):
            one_result = self.label_name_dict[ind]
            one_result['probability'] = prob
            results.append(one_result)     
        return {"status": status, "message": message, "results": results}
        