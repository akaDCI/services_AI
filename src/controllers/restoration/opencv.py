from typing import Dict
from PIL import Image
import cv2 as cv
from ._base import BaseRestorationProvider


class OpenCVRestorationProvider(BaseRestorationProvider):
    def __init__(self, config: Dict):
        super().__init__(config)

    def infer(self, src, mask):
        if isinstance(src, Image.Image):
            src = self.pil_to_numpy(src)
        if isinstance(mask, Image.Image):
            mask = cv.cvtColor(self.pil_to_numpy(mask), cv.COLOR_RGB2GRAY)
        else:
            mask = cv.cvtColor(mask, cv.COLOR_RGB2GRAY)

        return cv.inpaint(src, mask, 3, cv.INPAINT_NS)
