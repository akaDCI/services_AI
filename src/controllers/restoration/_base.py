from abc import abstractmethod
from typing import Dict, Union, Literal
from PIL import Image
import numpy as np
import numpy.typing as npt


_ModelOutputFormat = Literal["pillow", "numpy"]


class BaseRestorationProvider:
    def __init__(self, config: Dict, output_format: _ModelOutputFormat = "pillow"):
        """
        Initialize model, variables, configs
        """
        self.config = config
        self.output_format = output_format

    @abstractmethod
    def infer(
        self,
        src: Union[Image.Image, npt.NDArray[np.uint8]],
        mask: Union[Image.Image, npt.NDArray[np.uint8]]
    ) -> npt.NDArray:
        """
        Inference method of inpainting.

        Example:
            ```
            model = RestorationProvider()
            result_as_pil = model.infer(image, image_mask)
            ```
        """

    def pil_to_numpy(self, src: Image.Image):
        return np.array(src)

    def numpy_to_pil(self, src: npt.NDArray):
        return Image.fromarray(src, mode="RGB")
