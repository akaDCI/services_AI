import os
import io
import uuid
import shutil
import numpy as np
from PIL import Image


StaticDirectory = "data"


def create_directory(path: str) -> str:
    _path = os.path.join(os.getcwd(), path)
    os.makedirs(_path, exist_ok=True)
    return _path


def create_with_directory(directory: str) -> str:
    return create_directory(os.path.join(StaticDirectory, directory))


def save_file(directory: str, file: io.BytesIO, file_ext: str, expiration=None) -> str:
    _path = os.path.join(StaticDirectory, directory)
    create_directory(_path)
    _file_name = f"{uuid.uuid4().hex}.{file_ext}"
    _file_path = os.path.join(_path, _file_name)
    with open(_file_path, 'wb') as f:
        shutil.copyfileobj(file, f)
    return _file_path


def save_image(directory: str, image: Image.Image, expiration=None) -> str:
    _path = os.path.join(StaticDirectory, directory)
    create_directory(_path)
    _ext = image.format.lower() if image.format else 'png'
    _file_name = f"{uuid.uuid4().hex}.{_ext}"
    _file_path = os.path.join(_path, _file_name)
    image.save(_file_path)
    return _file_path


def save_images(directory: str, images: list[Image.Image], expiration=None) -> list[str]:
    _paths = []
    for image in images:
        _paths.append(save_image(directory, image, expiration))
    return _paths


def loads_static(paths: list[str], mode="RGB", type="pil") -> list[Image.Image] | list[np.ndarray]:
    _imgs = [Image.open(os.path.join(os.getcwd(), path)).convert(mode)
             for path in paths]
    if type == "np":
        return [np.array(img) for img in _imgs]
    return _imgs
