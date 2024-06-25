import time
import logging
from .opencv import OpenCVRestorationProvider


class RestorationController:
    def __init__(self, provider: str = "opencv"):
        self.provider = provider
        self.model = self.__get_provider(provider)

    def __get_provider(self, provider: str):
        if provider == "opencv":
            return OpenCVRestorationProvider
        elif provider == "lama":
            pass
        elif provider == "diffusion":
            pass
        else:
            raise ValueError(f"Provider {provider} is invalid!")

    def infer(self, src, mask):
        s = time.time()
        inpainted = self.model.infer(src, mask)
        logging.info(
            f"Inferred {self.__class__.__name__} [{round(time.time() - s)}s]")

        return inpainted
