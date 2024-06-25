from typing import Union
import numpy.typing as npt
import os
import uuid
import logging
from PIL import Image
import cv2 as cv
import numpy as np


TEMP_DIRECTORY = "temp"


class Temper:
    """
    Help save data to temp folder
    """
    @staticmethod
    def get_temp_dir(directory=TEMP_DIRECTORY) -> str:
        _path = os.path.join(os.getcwd(), directory)
        if os.path.exists(_path):
            return _path
        os.makedirs(_path, exist_ok=True)
        return _path

    @staticmethod
    def save_pwd_image(
        data: Union[Image.Image, npt.NDArray, cv.Mat],
        prefix: str = TEMP_DIRECTORY,
        ext: str = "jpg"
    ) -> str:
        _id = f"{prefix}_{uuid.uuid4().hex}.{ext}"
        _path = os.path.join(Temper.get_temp_dir(), _id)

        # Write to folder
        if isinstance(data, Image.Image):
            data.save(_path)
        elif isinstance(data, np.ndarray) or isinstance(data, cv.Mat):
            cv.imwrite(_path, data)
        else:
            raise ValueError(f"Unsupport types {type(data)}")

        logging.info(f"Saved {_path}")

        return f"{TEMP_DIRECTORY}/{_id}"
