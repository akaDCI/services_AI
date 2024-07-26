import uuid
import os
import io
import traceback
from dataclasses import dataclass, field
from PIL import Image
from typing import Annotated, List
from fastapi import FastAPI, Request, Response, UploadFile, File, Form, Depends, HTTPException, status
from fastapi.responses import RedirectResponse, ORJSONResponse, StreamingResponse
from .controllers.restoration import RestorationController, InferenceProvider, InferenceServer
from .controllers.crack_detection import CrackSegController
from .controllers.llm import LLMInputs, LLMController
from src.utils.static import save_images, save_file
from src.utils.client import get_client, Client
from src.utils.response import ResponseData


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
            provider="default")  # default or unet or yolo
        self.llm = LLMController()

        # Register routes
        self.app.get("/")(self.main)
        self.app.post("/infer")(self.infer)
        self.app.post("/uploads")(self.uploads)
        self.app.post("/api/crack_seg")(self.crackseg_infer)
        self.app.post("/api/restore")(self.restoration_infer)
        self.app.post("/api/llm")(self.chat_llm)

    async def main(self):
        """
        Redirect to the Swagger documents
        """
        return RedirectResponse("/docs")

    async def uploads(
        self,
        upload_files: Annotated[List[UploadFile], File(...)],
        client: Annotated[Client, Depends(get_client)]
    ):
        """
        Uploads
        """
        _paths = []

        # Iterate over upload files
        for upload_file in upload_files:
            # Read file extension
            file_ext = upload_file.filename.split(".")[-1]
            # Save file
            _paths.append(save_file("uploads", upload_file.file, file_ext))

        _response = {
            "uploads": _paths
        }
        client.update(_response)
        client.save()

        return ResponseData(_response)

    async def crackseg_infer(self, upload_images: list[UploadFile] = File(...)):
        """
        Crack segmentation
        """
        name_folder = uuid.uuid4().hex
        folder_path = f"tmp/upload_files/{name_folder}"
        # create folder
        os.makedirs(folder_path, exist_ok=True)
        for image in upload_images:
            img_filename = image.filename.split(".")[-1]
            if img_filename.lower() not in ["jpg", "jpeg", "png"]:
                raise HTTPException(
                    status_code=400, detail=f"Invalid image format {image.filename}")

            name_image = uuid.uuid4().hex
            # save image in folder_path
            image_path = f'{folder_path}/{name_image}.jpg'
            try:
                image_data = await image.read()
                pil_image = Image.open(io.BytesIO(image_data))
                rgb_pil_image = pil_image.convert('RGB')
                rgb_pil_image.save(image_path)
            except Exception as e:
                raise HTTPException(
                    status_code=400, detail=f"Failed to process image {image.filename}: {str(e)}")
        seg_results, raw_imgs, pred_imgs = self.crack_seg_infer.infer(
            name_folder)
        return {"msg": "Success",
                "seg_results":  seg_results}

    async def restoration_infer(
        self,
        images: Annotated[List[UploadFile], File(...)],
        masks: Annotated[List[UploadFile], File(...)],
        stream: Annotated[bool, Form()] = False,
        provider: Annotated[InferenceProvider, Form()] = "crfill",
        server: Annotated[InferenceServer, Form()] = "torch"
    ):
        """
        Crack restoration
        """
        # Read images, masks data
        _images, _masks = [], []
        for image, mask in zip(images, masks):
            _images.append(await image.read())
            _masks.append(await mask.read())

        # Inference
        try:
            inpainteds = self.restoration.infer(
                _images,
                _masks,
                provider,
                server,
                "bytes" if stream else "pillow"
            )
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            )

        # If stream is True, return Streaming response
        if stream:
            def stream_iteration():
                for inpainted in inpainteds:
                    buffer = io.BytesIO(inpainted)
                    yield buffer
            return StreamingResponse(stream_iteration(), media_type="image/png")

        # Else, save file locally and return path
        _paths = save_images("restore", inpainteds)
        return ORJSONResponse({
            "paths": _paths
        })

        # if stream == False:
        #     return JSONResponse({
        #         "path": result
        #     })

    async def infer(
        self,
        images: Annotated[List[UploadFile], File(...)],
        restoration_provider: Annotated[InferenceProvider, Form()] = "crfill",
        restoration_server: Annotated[InferenceServer, Form()] = "torch"
    ):
        """
        Pipeline inference
        """
        # Crack detection
        name_folder = uuid.uuid4().hex
        folder_path = f"tmp/upload_files/{name_folder}"

        # create folder
        os.makedirs(folder_path, exist_ok=True)
        for image in images:
            img_filename = image.filename.split(".")[-1]
            if img_filename.lower() not in ["jpg", "jpeg", "png"]:
                raise HTTPException(
                    status_code=400, detail=f"Invalid image format {image.filename}")

            name_image = uuid.uuid4().hex
            # save image in folder_path
            image_path = f'{folder_path}/{name_image}.jpg'
            try:
                image_data = await image.read()
                pil_image = Image.open(io.BytesIO(image_data))
                rgb_pil_image = pil_image.convert('RGB')
                rgb_pil_image.save(image_path)
            except Exception as e:
                raise HTTPException(
                    status_code=400, detail=f"Failed to process image {image.filename}: {str(e)}")
        seg_results, raw_imgs, pred_imgs = self.crack_seg_infer.infer(
            name_folder)

        # Crack inpaint
        inpainteds = self.restoration.infer(
            raw_imgs, pred_imgs, restoration_provider, restoration_server, return_type="pillow", preserved_color=True)
        inpaint_results = save_images("restoration", inpainteds)

        return {
            "msg": "Success",
            "seg_results":  seg_results,
            "inpaint_results": inpaint_results
        }

    async def chat_llm(self, data: LLMInputs):
        """
        Chat with LLM
        """
        try:
            # Set provider
            self.llm.set_provider(data.provider)

            # Infer based on stream or not
            if data.use_stream:
                result = self.llm.generate_async(
                    prompt=data.question, knowledge=data.knowledge)
                return StreamingResponse(result, media_type="text/plain")
            else:
                result = self.llm.generate(
                    prompt=data.question, knowledge=data.knowledge)
                return ORJSONResponse(content={
                    "msg": "Success",
                    "answer": result
                })
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e))

    @property
    def __call__(self):
        return self.app
