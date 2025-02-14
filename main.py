import os
import logging
import uvicorn
from fastapi.staticfiles import StaticFiles
from src.services import Services
from src.utils.static import StaticDirectory, create_directory
import gdown


class AIServer:
    def __init__(self):
        # Download model when initialize
        # unet_model_path = os.path.join(
        #     os.getcwd(), "models", "model_unet_vgg_16_best.pt")
        # if not os.path.exists(unet_model_path):
        #     gdown.download(id="1WfseljuUpMak1lLvyzRFIeDHSbgxv8Sn",
        #                    output=unet_model_path)
        # yolo_model_path = os.path.join(
        #     os.getcwd(), "models", "yolov8x_crack_seg.pt")
        # if not os.path.exists(yolo_model_path):
        #     gdown.download(id="1F-3ZAd1lluOT1quedjv2Xd00sVnSq92o",
        #                    output=yolo_model_path)

        # Initialize services
        self.api = Services()

        # Open temp folder for static file access
        self.api.app.mount(
            f"/{StaticDirectory}", StaticFiles(directory=create_directory(StaticDirectory)), name=StaticDirectory)

    def __call__(self):
        return self.api.app


if __name__ == "__main__":
    server = AIServer()
    logging.basicConfig(
        level=logging.INFO,
        format="%(filename)s (%(lineno)d) | \033[1m%(asctime)s\033[0m | \033[96m%(levelname)s\033[0m | %(message)s",
        datefmt='%d-%b-%y %H:%M:%S'
    )
    uvicorn.run(
        server,
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", 7860)),
        factory=True
    )
