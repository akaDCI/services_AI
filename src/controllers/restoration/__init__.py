from typing import List
import numpy.typing as npt
import time
import logging
from PIL import Image
from ._enum import *
from .opencv import OpenCVRestorationProvider
from .crfill import CRFillRestorationProvider
from .diffusion import DiffusionRestorationProvider


class RestorationController:
    def __init__(self, provider: str = "crfill"):
        self.model = self.__get_provider(provider)

    def __get_provider(self, provider: str):
        if provider == "crfill":
            return CRFillRestorationProvider()
        elif provider == "opencv":
            return OpenCVRestorationProvider()
        # elif provider == "diffusion":
        #     return DiffusionRestorationProvider()
        else:
            raise ValueError(f"Provider {provider} invalid")

    def set_provider(self, provider: str):
        self.model = self.__get_provider(provider)

    def infer(
        self,
        images: List[npt.NDArray],
        masks: List[npt.NDArray]
    ) -> List[Image.Image]:
        _s = time.time()
        inpainteds = self.model.infer(images, masks)
        logging.info(
            f"Inferred {self.model.__class__.__name__} [{round(time.time() - _s, 4)}s]")

        return [Image.fromarray(i).convert('RGB') for i in inpainteds]
