import time
import logging
import shutil
from src.controllers.crack_detection.unet import UnetCrackSeg
from src.controllers.crack_detection.yolo import YoloCrackSeg
class CrackSegController:
    def __init__(self, provider: str = "unet"):
        self.provider = provider
        self.model = self.__get_provider(provider)

    def __get_provider(self, provider: str):
        if provider == "unet":
            return UnetCrackSeg()
        elif provider == "yolo":
            return YoloCrackSeg()
        else:
            raise ValueError(f"Provider {provider} is invalid!")

    def infer(self, img_folder):
        img_dir = f"tmp/upload_files/{img_folder}"
        s = time.time()
        raw_imgs, pred_imgs = self.model.infer(img_folder)
        shutil.rmtree(img_dir)
        logging.info(
            f"Inferred {self.__class__.__name__} [{round(time.time() - s)}s]")

        return raw_imgs, pred_imgs