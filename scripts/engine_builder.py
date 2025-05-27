

import tensorrt as trt
import os
import onnx

from PIL import Image as Image

class EngineBuilder:
    """
    Parses an ONNX graph and builds a TensorRT engine from it.
    """

    def __init__(self,
                 onnx_model_path,
                 output_engine_path,
                 shape = None,
                 dla_core = -1,
                 verbose=False,
    ):
        """
        :param onnx_model_name: Onnx model path where to create Tensorrt engine from
        :param output_engine_path: The path where to serialize the engine to.
        :param shape: Expected input shape for Tensorrt engine
        :param verbose: If enabled, a higher verbosity level will be set on the TensorRT logger.
        """

        print("Loading the ONNX file...")
        onnx_data = self.load_onnx(onnx_model_path)
        if onnx_data is None:
            return RuntimeError("Onnx model can not be loaded.")
        
        if shape == None:
            onnx_model = onnx.load(onnx_model_path)
            self.shape = [[d.dim_value for d in _input.type.tensor_type.shape.dim] for _input in onnx_model.graph.input][0]
        else:
            self.shape = shape
        
        MAX_BATCH_SIZE = self.shape[0]

        TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger()
        self.EXPLICIT_BATCH = (
            []
            if trt.__version__[0] < "7"
            else [1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)]
        )

        with trt.Builder(TRT_LOGGER) as self.builder, self.builder.create_network(
            *self.EXPLICIT_BATCH
        ) as self.network, trt.OnnxParser(self.network, TRT_LOGGER) as self.parser:

            if not self.parser.parse(onnx_data):
                print("ERROR: Failed to parse the ONNX file.")
                for error in range(self.parser.num_errors):
                    print(self.parser.get_error(error))
                return None

        print("Building the TensorRT engine.  This would take a while...")
        print('(Use "--verbose" or "-v" to enable verbose logging.)')

        self.builder.max_batch_size = MAX_BATCH_SIZE
        self.config = self.builder.create_builder_config()
        self.config.max_workspace_size = 1 << 30
        self.config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
        self.profile = self.builder.create_optimization_profile()
        tensor_shape = self.network.get_input(0).shape
        
        self.profile.set_shape(
            self.network.get_input(0).name,  # input tensor name
            (MAX_BATCH_SIZE, tensor_shape[1], tensor_shape[2], tensor_shape[3]),  # min shape
            (MAX_BATCH_SIZE, tensor_shape[1], tensor_shape[2], tensor_shape[3]),  # opt shape
            (MAX_BATCH_SIZE, tensor_shape[1], tensor_shape[2], tensor_shape[3]),  # max shape
        )  
        self.config.add_optimization_profile(self.profile)
        
        if not self.builder.platform_has_fast_fp16:
                RuntimeError("FP16 is not supported natively on this platform/device")
        else:
            self.config.set_flag(trt.BuilderFlag.FP16)

       
        
        self.engine = self.builder.build_engine(self.network, self.config)
        if self.engine is not None:
            print("Completed creating engine.")
        with open(output_engine_path, "wb") as f:
            f.write(self.engine.serialize())
        print("Serialized the TensorRT engine to file: %s" % output_engine_path)
        print("Pytorch model is serialized.")
                    
    def get_engine(self):
        if self.engine is not None:
            return self.engine
        else:
            raise SystemExit("ERROR: failed to return the TensorRT engine!")
    
    def load_onnx(self, onnx_model_path):
        """Read the ONNX file."""
        onnx_path = "%s" % onnx_model_path
        if not os.path.isfile(onnx_path):
            print(
                "ERROR: file (%s) not found!"
                % onnx_path
            )
            return None
        else:
            with open(onnx_path, "rb") as f:
                return f.read()