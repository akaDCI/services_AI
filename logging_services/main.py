import os
import logging
import uvicorn
from fastapi.staticfiles import StaticFiles
from src.services import Services


class AIServer:
    def __init__(self):
        self.api = Services()

        self.api.app.mount(
            f"/report", StaticFiles(directory="/Users/mac/Documents/akaDCI/akaDCI_hotdev/services_AI/src/logging_services/visual"), name="report")

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
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", 8000)),
        factory=True
    )
