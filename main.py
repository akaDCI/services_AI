import logging.config
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
    logging.log(level=logging.INFO, msg="Starting server 🐧")
    uvicorn.run(
        server,
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", 7860)),
        factory=True
    )
