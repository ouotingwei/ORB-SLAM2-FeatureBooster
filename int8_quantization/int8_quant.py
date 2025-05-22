import tensorrt as trt
import numpy as np
import os
import glob
import pycuda.driver as cuda
import pycuda.autoinit

class FeatureBoosterCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calib_dir, max_samples=100):
        super().__init__()
        self.file_list = sorted(glob.glob(os.path.join(calib_dir, "*.npz")))[:max_samples]
        self.current_index = 0
        self.batch_size = 1

        self.desc_shape = (1, 8000, 256)
        self.kps_shape  = (1, 8000, 4)

        self.desc_dev = cuda.mem_alloc(int(np.prod(self.desc_shape) * 4))
        self.kps_dev  = cuda.mem_alloc(int(np.prod(self.kps_shape) * 4))

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index >= len(self.file_list):
            return None

        data = np.load(self.file_list[self.current_index])
        desc = data["descriptors"]
        kps  = data["keypoints"]

        N = desc.shape[0]
        desc_padded = np.zeros(self.desc_shape, dtype=np.float32)
        kps_padded  = np.zeros(self.kps_shape, dtype=np.float32)
        desc_padded[0, :N, :] = desc
        kps_padded[0, :N, :]  = kps

        cuda.memcpy_htod(self.desc_dev, desc_padded)
        cuda.memcpy_htod(self.kps_dev, kps_padded)

        self.current_index += 1
        return [int(self.desc_dev), int(self.kps_dev)]

    def read_calibration_cache(self):
        if os.path.exists("calib.table"):
            with open("calib.table", "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open("calib.table", "wb") as f:
            f.write(cache)

# ========== Build INT8 Engine ==========
TRT_LOGGER = trt.Logger(trt.Logger.INFO)

onnx_path      = "/feature_booster_explicit.onnx"
engine_path    = "feature_booster_int8.engine"
calib_data_dir = "/calib_data"

builder = trt.Builder(TRT_LOGGER)
network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network = builder.create_network(network_flags)
parser = trt.OnnxParser(network, TRT_LOGGER)

with open(onnx_path, "rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError("Failed to parse ONNX model")

config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.INT8)
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

# dynamic shape
profile = builder.create_optimization_profile()
profile.set_shape("descriptors", (1, 1, 256), (1, 4000, 256), (1, 8000, 256))
profile.set_shape("keypoints",   (1, 1, 4),   (1, 4000, 4),   (1, 8000, 4))
config.add_optimization_profile(profile)

# INT8 calibrator
calibrator = FeatureBoosterCalibrator(calib_data_dir, max_samples=100)
config.int8_calibrator = calibrator

serialized_engine = builder.build_serialized_network(network, config)
if serialized_engine is None:
    raise RuntimeError("Failed to build INT8 serialized engine")

with open(engine_path, "wb") as f:
    f.write(serialized_engine)

