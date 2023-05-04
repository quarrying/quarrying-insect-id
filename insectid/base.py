import onnxruntime
import numpy as np


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
            

def check_image_dtype_and_shape(image):
    if not isinstance(image, np.ndarray):
        raise Exception(f'image is not np.ndarray!')

    if isinstance(image.dtype, (np.uint8, np.uint16)):
        raise Exception(f'Unsupported image dtype, only support uint8 and uint16, got {image.dtype}!')
    if image.ndim not in {2, 3}:
        raise Exception(f'Unsupported image dimension number, only support 2 and 3, got {image.ndim}!')
    if image.ndim == 3:
        num_channels = image.shape[-1]
        if num_channels not in {1, 3, 4}:
            raise Exception(f'Unsupported image channel number, only support 1, 3 and 4, got {num_channels}!')

