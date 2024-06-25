from dataclasses import dataclass, field
from fastapi import FastAPI, Request, Response
from fastapi.responses import RedirectResponse
from .middlewares.body_parser import BodyParserMiddleware


@dataclass
class Services:
    """API Services"""
    app: "FastAPI" = field(default_factory=FastAPI)
    request: Request = field(default=None)
    response: Response = field(default=None)

    # define router here
    def __post_init__(self):
        """Post init"""
        # Middleware
        self.app.add_middleware(BodyParserMiddleware)

        # Register routes
        self.app.get("/")(self.main)

    async def main(self, request: Request, response: Response):
        """
        Redirect to the Swagger documents
        """
        return RedirectResponse("/docs")

    @property
    def __call__(self):
        return self.app
