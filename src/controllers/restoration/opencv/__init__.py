import logging
import numpy as np
import cv2 as cv
from .._base import BaseRestorationProvider, BaseConfig


class DefaultConfig(BaseConfig):
    inpaintRadius = 3
    flags = cv.INPAINT_NS


class OpenCVRestorationProvider(BaseRestorationProvider):
    def __init__(self, config=DefaultConfig):
        super().__init__(config)

    def infer(self, images, masks, server):
        logging.warning(
            "Server option is not available in OpenCV restoration.")

        inpainteds = []
        for image, mask in zip(images, masks):
            inpainted = cv.inpaint(
                image, mask, self.config.inpaintRadius, self.config.flags)
            inpainteds.append(np.array(inpainted))

        return inpainteds
