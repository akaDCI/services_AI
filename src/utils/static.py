import os
import uuid
from PIL import Image


StaticDirectory = "data"


def create_directory(path: str) -> str:
    _path = os.path.join(os.getcwd(), path)
    os.makedirs(_path, exist_ok=True)
    return _path


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
