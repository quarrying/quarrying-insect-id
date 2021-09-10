import multiprocessing
import onnxruntime


class OnnxModel(object):
    def __init__(self, model_path):
        sess_options = onnxruntime.SessionOptions()
        # # Set graph optimization level to ORT_ENABLE_EXTENDED to enable bert optimization.
        # sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        # # Use OpenMP optimizations. Only useful for CPU, has little impact for GPUs.
        # sess_options.intra_op_num_threads = multiprocessing.cpu_count()

        self.sess = onnxruntime.InferenceSession(model_path, sess_options)
        self._input_names = [item.name for item in self.sess.get_inputs()]
        self._output_names = [item.name for item in self.sess.get_outputs()]
        
    @property
    def input_names(self):
        return self._input_names
        
    @property
    def output_names(self):
        return self._output_names
        
    def forward(self, inputs):
        if not isinstance(inputs, (tuple, list)):
            inputs = [inputs]
        input_feed = {name: input for name, input in zip(self.input_names, inputs)}
        outputs = self.sess.run(self.output_names, input_feed)
        if len(self.output_names) == 1:
            return outputs[0]
        else:
            return outputs
            

        