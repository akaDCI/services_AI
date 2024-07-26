import time
import logging
import shutil
from src.controllers.crack_detection.unet import UnetCrackSeg
from src.controllers.crack_detection.yolo import YoloCrackSeg
from src.controllers.crack_detection.seg_former import FormerCrackSeg


class CrackSegController:
    def __init__(self, provider: str = "segformer"):
        self.provider = provider
        self.model = self.__get_provider(provider)

    def __get_provider(self, provider: str = "segformer"):
        if provider == "segformer":
            return FormerCrackSeg()
        # elif provider == "yolo":
        #     return YoloCrackSeg()
        # elif provider == "unet":
        #     return UnetCrackSeg()
        else:
            raise ValueError(f"Provider {provider} invalid")

    def set_provider(self, provider: str):
        self.model = self.__get_provider(provider)

    def infer(self, images, threshold: float = 0.65):
        s = time.time()
        crack_results = self.model.infer(images, threshold)
        logging.info(
            f"Inferred {self.__class__.__name__} [{round(time.time() - s)}s]")

        return crack_results
