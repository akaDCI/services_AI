import uuid
import os
import io
from PIL import Image
from typing import Annotated
from typing import Annotated
from dataclasses import dataclass, field
from fastapi import FastAPI, Request, Response, UploadFile, File, Form, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse, StreamingResponse
from .controllers.restoration import RestorationController
from .controllers.crack_detection import CrackSegController


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
        self.crack_seg_infer = CrackSegController(
            provider="unet")  # unet(default) or yolo

        # Register routes
        self.app.get("/")(self.main)
        self.app.post("/infer")(self.infer)
        self.app.post("/api/restore")(self.restoration_infer)
        self.app.post("/api/crack_seg")(self.crackseg_infer)

    async def main(self, request: Request, response: Response):
        """
        Redirect to the Swagger documents
        """
        return RedirectResponse("/docs")

    async def restoration_infer(
        self,
        image: Annotated[UploadFile, File(...)],
        mask: Annotated[UploadFile, File(...)],
        stream: Annotated[bool, Form()] = False
    ):
        """
        Crack restoration
        """
        _image = await image.read()
        _mask = await mask.read()

        result = self.restoration.infer(
            _image, _mask, True if stream == False else False)

        if stream == False:
            return JSONResponse({
                "path": result
            })

        return StreamingResponse(result, media_type="application/octet-stream", headers={"Content-Disposition": f"attachment;filename={image.filename}"})

    async def crackseg_infer(self, upload_images: list[UploadFile] = File(...)):
        """
        Crack segmentation
        """
        name_folder = uuid.uuid4().hex
        folder_path = f"tmp/upload_files/{name_folder}"
        # create folder
        os.makedirs(folder_path, exist_ok=True)
        for image in upload_images:
            name_image = uuid.uuid4().hex
            # save image in folder_path
            image_path = f'{folder_path}/{name_image}.jpg'
            try:
                image_data = await image.read()
                pil_image = Image.open(io.BytesIO(image_data))
                pil_image.save(image_path)
            except Exception as e:
                raise HTTPException(
                    status_code=400, detail=f"Failed to process image {image.filename}: {str(e)}")
        raw_imgs, pred_imgs = self.crack_seg_infer.infer(name_folder)
        return {"msg": "Success"}

    async def infer(self, upload_images: list[UploadFile] = File(...)):
        """
        Pipeline inference
        """
        # Crack detection
        name_folder = uuid.uuid4().hex
        folder_path = f"tmp/upload_files/{name_folder}"
        # create folder
        os.makedirs(folder_path, exist_ok=True)
        for image in upload_images:
            name_image = uuid.uuid4().hex
            # save image in folder_path
            image_path = f'{folder_path}/{name_image}.jpg'
            try:
                image_data = await image.read()
                pil_image = Image.open(io.BytesIO(image_data))
                pil_image.save(image_path)
            except Exception as e:
                raise HTTPException(
                    status_code=400, detail=f"Failed to process image {image.filename}: {str(e)}")
        raw_imgs, pred_imgs = self.crack_seg_infer.infer(name_folder)

        # Crack inpaint
        results = []
        for _img, _mask in zip(raw_imgs, pred_imgs):
            print(_img.dtype, _mask.dtype)
            results.append(self.restoration.infer(_img, _mask, True))

        return {
            "paths": results
        }

    @property
    def __call__(self):
        return self.app
