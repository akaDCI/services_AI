from typing import Union, List
import numpy.typing as npt
import io
import time
import logging
import numpy as np
import cv2 as cv
from .opencv import OpenCVRestorationProvider
from src.utils.temp import Temper


class RestorationController:
    def __init__(self, provider: str = "opencv"):
        self.provider = provider
        self.model = self.__get_provider(provider)

    def __get_provider(self, provider: str):
        if provider == "opencv":
            return OpenCVRestorationProvider({})
        elif provider == "lama":
            pass
        elif provider == "diffusion":
            pass
        else:
            raise ValueError(f"Provider {provider} is invalid!")

    def infer(
        self,
        src: Union[bytes, npt.NDArray],
        mask: Union[bytes, npt.NDArray],
        save: bool = False,
        ext: str = "jpeg"
    ):
        # Convert to array
        if isinstance(src, bytes):
            src = cv.imdecode(np.frombuffer(src, np.uint8), -1)
        if isinstance(mask, bytes):
            mask = cv.imdecode(np.frombuffer(mask, np.uint8), -1)

        # Check shape of mask
        is_mask_repair = False
        if len(mask.shape) != 3:
            is_mask_repair = True

        s = time.time()
        inpainted = self.model.infer(src, mask, is_mask_repair)
        logging.info(
            f"Inferred {self.__class__.__name__} [{round(time.time() - s, 4)}s]")

        if save:
            return Temper.save_pwd_image(inpainted, prefix="restore", ext=ext)

        buffer = io.BytesIO()
        img = self.model.numpy_to_pil(inpainted)
        img.save(buffer, format=ext)
        return buffer

    def infers(
        self,
        src: List[Union[bytes, npt.NDArray]],
        mask: List[Union[bytes, npt.NDArray]],
        save: bool = False,
        ext: str = "jpeg"
    ):
        for _src, _mask in zip(src, mask):
            yield self.infer(_src, _mask, save, ext)
