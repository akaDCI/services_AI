import os
import logging
import uvicorn
from src.services import Services


class AIServer:
    def __init__(self):
        self.api = Services()

    def __call__(self):
        return self.api.app


if __name__ == "__main__":
    server = AIServer()
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(name)s, %(filename)s (%(lineno)d) | \033[1m%(asctime)s\033[0m | \033[96m%(levelname)s\033[0m | %(message)s",
        datefmt='%d-%b-%y %H:%M:%S'
    )
    uvicorn.run(
        server,
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", 7860)),
        factory=True
    )
