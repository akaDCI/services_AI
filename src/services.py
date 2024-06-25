from typing import Annotated
from dataclasses import dataclass, field
from fastapi import FastAPI, Request, Response, UploadFile, File
from fastapi.responses import RedirectResponse
from .controllers.restoration import RestorationController


@dataclass
class Services:
    """API Services"""
    app: "FastAPI" = field(default_factory=FastAPI)
    request: Request = field(default=None)
    response: Response = field(default=None)

    # define router here
    def __post_init__(self):
        """Post init"""
        # Intialize services
        self.restoration = RestorationController()

        # Register routes
        self.app.get("/")(self.main)
        self.app.post("/api/restore")(self.restoration_infer)

    async def main(self, request: Request, response: Response):
        """
        Redirect to the Swagger documents
        """
        return RedirectResponse("/docs")

    async def restoration_infer(self, image: Annotated[UploadFile, File(...)]):
        """
        Crack restoration
        """
        data = await image.read()
        print(len(data))
        return {"hello": "hi"}

    @property
    def __call__(self):
        return self.app
