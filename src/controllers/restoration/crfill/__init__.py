import torch
import numpy as np
from .._base import BaseRestorationProvider, BaseConfig
from .._enum import InferenceServer
from .model import BaseConvGenerator
from src.utils.torch_infer import torch_inference
from src.utils.onnx_infer import onnx_interence_session, onnx_inference
from src.utils.downloader import download_model_from_drive


class DefaultConfig(BaseConfig):
    torch_model_id = "1Mr_AysRFFim5BHeQOhO9zZkuZ2eMAaBR"
    torch_model_name = "crfill.pth"
    onnx_model_id = "1jjXsN5p4gWKXm_xuW5MlxkLASxygw470"
    onnx_model_name = "crfill.onnx"


class CRFillRestorationProvider(BaseRestorationProvider):
    def __init__(self, config=DefaultConfig):
        super().__init__(config)
        self.model = self.__download_model()

    def __download_model(self):
        _model = BaseConvGenerator()
        # Download torch model
        _model_path = download_model_from_drive(
            self.config.torch_model_id, self.config.torch_model_name)
        # Load torch checkpoint
        _model_state = torch.load(_model_path)
        _model.load_state_dict(_model_state)
        # Download onnx model
        # _model_path = download_model_from_drive(
        #     self.config.onnx_model_id, self.config.onnx_model_name)
        # self.onnx_session = onnx_interence_session(_model_path)
        return _model

    def infer(self, images, masks):
        # Preprocessing image, mask
        imgs = torch.Tensor(
            np.array(images)).div(255.).permute(0, 3, 1, 2)  # Shape [b, 3, h, w]
        mks = torch.Tensor(np.array(masks)).unsqueeze(1)  # Shape [b, 1, h, w]

        # Inference
        inpainteds = torch_inference(self.model, imgs, mks)

        # Postprocessing image
        inpainteds = inpainteds.numpy().astype(np.uint8)
        return [i.transpose((1, 2, 0)) for i in inpainteds]

        # if server == InferenceServer.Torch:
        #     # Preprocessing image, mask
        #     imgs = torch.Tensor(
        #         np.array(images)).div(255.).permute(0, 3, 1, 2)  # Shape [b, 3, h, w]
        #     mks = torch.Tensor(np.array(masks)).unsqueeze(
        #         1)  # Shape [b, 1, h, w]

        #     # Inference
        #     inpainteds = torch_inference(self.model, imgs, mks)

        #     # Postprocessing image
        #     inpainteds = inpainteds.numpy().astype(np.uint8)
        #     return [i.transpose((1, 2, 0)) for i in inpainteds]

        # elif server == InferenceServer.Onnx:
        #     # Preprocessing
        #     imgs = (np.array(images).transpose(
        #         0, 3, 1, 2) / 255).astype(np.float32)
        #     mks = np.expand_dims(np.array(masks), axis=1).astype(np.float32)

        #     # Inference
        #     inpainteds = onnx_inference(self.onnx_session, imgs, mks)

        #     # Postprocessing
        #     inpainteds = inpainteds.astype(np.uint8)
        #     return [i.transpose((1, 2, 0)) for i in inpainteds]

        # elif server == InferenceServer.Triton:
        #     raise NotImplementedError("Triton in CRFill is not implemented")

        # else:
        #     raise ValueError(f"Undefined Inference server!")
