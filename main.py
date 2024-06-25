import os
import logging
import uvicorn
from fastapi.staticfiles import StaticFiles
from src.services import Services
from src.utils.temp import TEMP_DIRECTORY


class AIServer:
    def __init__(self):
        self.api = Services()

        # Open temp folder for static file access
        self.api.app.mount(
            f"/{TEMP_DIRECTORY}", StaticFiles(directory=os.path.join(os.getcwd(), TEMP_DIRECTORY)), name=TEMP_DIRECTORY)

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
