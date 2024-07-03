import enum


class InferenceProvider(enum.Enum):
    OpenCV = "opencv"
    CRFill = "crfill"
    LaMa = "lama"
    Diffusion = "diffusion"
    All = "all"


class InferenceServer(enum.Enum):
    Torch = "torch"
    Onnx = "onnx"
    Triton = "triton"
