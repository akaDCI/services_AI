import orjson
from starlette.middleware.base import BaseHTTPMiddleware


class BodyParserMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        body = request.body()

        # Parse data
        try:
            data = orjson.loads(body)
            request.state.body = data
        except orjson.JSONDecodeError:
            pass

        # Call the next endpoint
        return await call_next(request)
