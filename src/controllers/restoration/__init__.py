from typing import Union, List, Literal, Dict
import numpy.typing as npt
import time
import logging
import numpy as np
import cv2 as cv
from PIL import Image
from ._base import BaseRestorationProvider
from ._enum import *
from .opencv import OpenCVRestorationProvider
from .crfill import CRFillRestorationProvider
from .lama import LaMaRestorationProvider
from .diffusion import DiffusionRestorationProvider


class RestorationController:
    """
    ### Crack restoration (Inpainting)
    >>> model = RestorationController(late_init=True)

    Args:
        `late_init`: Download model when `infer()` is called. Else, download all models when initialize.

    >>> generated = model.infer(image, mask, provider="crfill", server="triton", return_type="pillow")
    >>> Image.fromarray(generated)

    Args:
        `image`: Original Image
        `mask`: Black & White inpaint region
        `provider`: Model that be used to inpaint
        `server`: Inference server that be used.
        `return_type`: Type of image be return. (`bytes`, `array`, `pillow`)

    Provider:
        `opencv`: Use `cv.inpaint()` method
        `crfill`: Use GANs model
        `lama`: Use Fourier UNet model
        `diffusion`: Use Diffusion model

    Server:
        `torch`: Use pytorch inference mode with `.pt` checkpoint
        `onnx`: Use ONNX inference with `.onnx` checkpoint
        `triton`: Call remote NVIDIA Triton Inference Server
    """

    def __init__(self, late_init: bool = False):
        self.providers: Dict[str, BaseRestorationProvider] = {}
        if not late_init:
            self.__get_provider(InferenceProvider.All)

    def __get_provider(self, provider: InferenceProvider) -> BaseRestorationProvider:
        _provider = None

        if provider == InferenceProvider.OpenCV or provider == InferenceProvider.All:
            if InferenceProvider.OpenCV.value not in self.providers:
                self.providers[InferenceProvider.OpenCV.value] = OpenCVRestorationProvider(
                )
                self.providers[InferenceProvider.OpenCV.value].initialize()
            _provider = self.providers[InferenceProvider.OpenCV.value]

        if provider == InferenceProvider.CRFill or provider == InferenceProvider.All:
            if InferenceProvider.CRFill.value not in self.providers:
                self.providers[InferenceProvider.CRFill.value] = CRFillRestorationProvider(
                )
                self.providers[InferenceProvider.CRFill.value].initialize()
            _provider = self.providers[InferenceProvider.CRFill.value]

        if provider == InferenceProvider.LaMa or provider == InferenceProvider.All:
            if InferenceProvider.LaMa.value not in self.providers:
                self.providers[InferenceProvider.LaMa.value] = LaMaRestorationProvider(
                )
                self.providers[InferenceProvider.LaMa.value].initialize()
            _provider = self.providers[InferenceProvider.LaMa.value]

        if provider == InferenceProvider.Diffusion or provider == InferenceProvider.All:
            if InferenceProvider.Diffusion.value not in self.providers:
                self.providers[InferenceProvider.Diffusion.value] = DiffusionRestorationProvider(
                )
                self.providers[InferenceProvider.Diffusion.value].initialize()
            _provider = self.providers[InferenceProvider.Diffusion.value]

        if _provider is None:
            raise ValueError(
                f"Invalid provider. Expected in {[e.value for e in InferenceProvider]}")

        return _provider

    def __array_to_bytes(self, inpainteds) -> List[bytes]:
        """
        Convert a list of NumPy arrays to a list of byte arrays.

        :param inpainteds: List of NumPy arrays.
        :return: List of byte arrays.
        """
        byte_arrays = []
        for array in inpainteds:
            # Encode the NumPy array to bytes using imencode
            _, buffer = cv.imencode('.png', array)
            byte_array = buffer.tobytes()
            byte_arrays.append(byte_array)
        return byte_arrays

    def __array_to_pillow(self, inpainteds, preserved_color=False) -> List[Image.Image]:
        """
        Convert a list of NumPy arrays to a list of PIL images.

        :param inpainteds: List of NumPy arrays.
        :return: List of PIL Images.
        """
        pil_images = []
        for array in inpainteds:
            # Convert NumPy array to PIL Image
            if not preserved_color:
                array = cv.cvtColor(array, cv.COLOR_BGR2RGB)
            image = Image.fromarray(array)
            pil_images.append(image)
        return pil_images

    def infer(
        self,
        images: List[Union[bytes, npt.NDArray, Image.Image]],
        masks: List[Union[bytes, npt.NDArray, Image.Image]],
        provider: InferenceProvider = InferenceProvider.CRFill,
        server: InferenceServer = InferenceServer.Torch,
        return_type: Literal['bytes', 'array', 'pillow'] = "array",
        **kwargs
    ) -> List[Union[bytes, npt.NDArray, Image.Image]]:
        # Convert to array
        _images, _masks = [], []
        for image, mask in zip(images, masks):
            # bytes -> array
            if isinstance(image, bytes):
                image = cv.imdecode(np.fromstring(
                    image, np.uint8), cv.IMREAD_COLOR)
            if isinstance(mask, bytes):
                mask = cv.imdecode(np.fromstring(
                    mask, np.uint8), cv.IMREAD_COLOR)
            # pil -> array
            if isinstance(image, Image.Image):
                image = np.array(image)
            if isinstance(mask, Image.Image):
                mask = np.array(mask)

            # Validate shape of image, mask
            if len(image.shape) != 3:
                if len(image.shape) == 4:
                    # Convert 4 dims to 3 dims image without pil
                    image = cv.cvtColor(image, cv.COLOR_RGBA2RGB)
                else:
                    raise ValueError(
                        f"Image must have 3 dimensions. Found {image.shape}")
            if len(mask.shape) != 2:
                if len(mask.shape) == 3:
                    # Convert to single channel mask
                    mask = cv.cvtColor(mask, cv.COLOR_RGB2GRAY)
                elif len(mask.shape) == 4:
                    mask = cv.cvtColor(mask, cv.COLOR_RGBA2GRAY)
                else:
                    raise ValueError(
                        f"Mask must have 2 dimension. Found {mask.shape}")

            _images.append(np.array(image))
            _masks.append(np.array(mask))

        # Get inference provider
        _model = self.__get_provider(provider)

        _s = time.time()
        inpainteds = _model.infer(_images, _masks, server)
        logging.info(
            f"Inferred {_model.__class__.__name__} [{round(time.time() - _s, 4)}s]")

        if return_type == "array":
            return inpainteds
        if return_type == "bytes":
            return self.__array_to_bytes(inpainteds)
        if return_type == "pillow":
            return self.__array_to_pillow(inpainteds, preserved_color=kwargs.get("preserved_color", False))
