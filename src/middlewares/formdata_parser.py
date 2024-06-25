from typing import Literal
import orjson
from PIL import Image
import numpy as np
from starlette.middleware.base import BaseHTTPMiddleware


_MediaOutputOptions = Literal["pillow", "numpy"]


class FormDataParserMiddleware(BaseHTTPMiddleware):
    def __init__(self, media_output: _MediaOutputOptions = "pillow"):
        self.media_output = media_output

    async def dispatch(self, request, call_next):
        if request.method == "POST" and request.headers.get("content-type", "").startswith("multipart/form-data"):
            form = await request.form()

            print("Form data", form)

        return await call_next(request)
