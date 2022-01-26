import multiprocessing

import cv2
import onnxruntime


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
    
    
class OnnxModel(object):
    def __init__(self, model_path):
        sess_options = onnxruntime.SessionOptions()
        # # Set graph optimization level to ORT_ENABLE_EXTENDED to enable bert optimization.
        # sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        # # Use OpenMP optimizations. Only useful for CPU, has little impact for GPUs.
        # sess_options.intra_op_num_threads = multiprocessing.cpu_count()
        onnx_gpu = (onnxruntime.get_device() == 'GPU')
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if onnx_gpu else ['CPUExecutionProvider']
        self.sess = onnxruntime.InferenceSession(model_path, sess_options, providers=providers)
        self._input_names = [item.name for item in self.sess.get_inputs()]
        self._output_names = [item.name for item in self.sess.get_outputs()]
        
    @property
    def input_names(self):
        return self._input_names
        
    @property
    def output_names(self):
        return self._output_names
        
    def forward(self, inputs):
        to_list_flag = False
        if not isinstance(inputs, (tuple, list)):
            inputs = [inputs]
            to_list_flag = True
        input_feed = {name: input for name, input in zip(self.input_names, inputs)}
        outputs = self.sess.run(self.output_names, input_feed)
        if (len(self.output_names) == 1) and to_list_flag:
            return outputs[0]
        else:
            return outputs
            
