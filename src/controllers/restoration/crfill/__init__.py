import torch
import numpy as np
from .._base import BaseRestorationProvider, BaseConfig
from .._enum import InferenceServer
from .model import BaseConvGenerator
from src.utils.torch_infer import torch_inference
from src.utils.downloader import download_model_from_drive


class DefaultConfig(BaseConfig):
    drive_id = "1Mr_AysRFFim5BHeQOhO9zZkuZ2eMAaBR"
    model_name = "crfill.pth"


class CRFillRestorationProvider(BaseRestorationProvider):
    def __init__(self, config=DefaultConfig):
        super().__init__(config)

    def initialize(self, *args, **kwargs):
        self.model = BaseConvGenerator()
        # Download model
        _model_path = download_model_from_drive(
            self.config.drive_id, self.config.model_name)
        # Load checkpoint
        _model_state = torch.load(_model_path)
        self.model.load_state_dict(_model_state)

    def infer(self, images, masks, server):
        if server == InferenceServer.Torch:
            # Preprocessing image, mask
            imgs = torch.Tensor(
                np.array(images)).div(255.).permute(0, 3, 1, 2)  # Shape [b, 3, h, w]
            mks = torch.Tensor(np.array(masks)).unsqueeze(
                1)  # Shape [b, 1, h, w]

            # Inference
            inpainteds = torch_inference(self.model, imgs, mks)

            # Postprocessing image
            inpainteds = inpainteds.numpy().astype(np.uint8)
            return [i.transpose((1, 2, 0)) for i in inpainteds]

        elif server == InferenceServer.Onnx:
            raise NotImplementedError("Onnx in CRFill is not implemented")

        elif server == InferenceServer.Triton:
            raise NotImplementedError("Triton in CRFill is not implemented")

        else:
            raise ValueError(f"Undefined Inference server!")
