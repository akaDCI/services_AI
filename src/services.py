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
from src.utils.static import save_images, save_file, loads_static
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
        self.crackseg = CrackSegController()
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

    async def crackseg_infer(
        self,
        client: Annotated[Client, Depends(get_client)],
        threshold: Annotated[float, Form(...)] = 0.65,
        provider: Annotated[str, Form(...)] = "segformer",
    ):
        """
        Crack segmentation
        """
        # Retrieve uploaded files
        _uploads = client.data.get("uploads", None)
        if not _uploads:
            raise HTTPException(
                status_code=400, detail="Uploads required. Do /uploads first.")

        # Load images
        _images = loads_static(_uploads)

        # Inference
        try:
            self.crackseg.set_provider(provider)
            _results = self.crackseg.infer(_images, threshold)
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            )

        # Save results
        _path = save_images("crackseg", _results)

        # Update client data
        _response = {
            "crackseg": _path
        }
        client.update(_response)
        client.save()

        return ResponseData(_response)

    async def restoration_infer(
        self,
        client: Annotated[Client, Depends(get_client)],
        provider: Annotated[str, Form(...)] = "crfill",
    ):
        """
        Crack restoration
        """
        # Retrieve uploaded files
        _images = client.data.get("uploads", None)
        _masks = client.data.get("crackseg", None)

        # Check if images and masks are available
        if not _images or not _masks:
            raise HTTPException(
                status_code=400, detail="Images and masks required. Do /uploads and /crack_seg first.")

        # Load images, masks
        _images = loads_static(_images, type="np")
        _masks = loads_static(_masks, mode="L", type="np")

        # Inference
        try:
            self.restoration.set_provider(provider)
            _results = self.restoration.infer(_images, _masks)
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            )

        # Save results
        _paths = save_images("restore", _results)

        # Update client data
        _response = {
            "restore": _paths
        }
        client.update(_response)
        client.save()

        return ResponseData(_response)

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
        seg_results, raw_imgs, pred_imgs = self.crackseg.infer(
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
                return ResponseData({
                    "answer": result
                })
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e))

    @ property
    def __call__(self):
        return self.app
