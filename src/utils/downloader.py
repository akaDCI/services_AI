"""
Download model to `models/`
"""
import logging
import os
import gdown


# Make dirs if models folder not exist
os.makedirs(os.path.join(os.getcwd(), "models"), exist_ok=True)


def download_model_from_drive(drive_id: str, name: str, force_redownload=False) -> str:
    """
    Download model from drive_id, name and return relative path
    """
    _model_path = os.path.join(os.getcwd(), "models", name)
    if force_redownload or not os.path.exists(_model_path):
        logging.info(f"Downloading {name} from Google drive")
        gdown.download(id=drive_id, output=_model_path)

    return _model_path
