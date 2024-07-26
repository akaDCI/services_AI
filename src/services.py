import traceback
from dataclasses import dataclass, field
from PIL import Image
from typing import Annotated, List
from fastapi import FastAPI, Request, Response, UploadFile, File, Form, Depends, HTTPException, status
from fastapi.responses import RedirectResponse, StreamingResponse
from .controllers.restoration import RestorationController
from .controllers.crack_detection import CrackSegController
from .controllers.llm import LLMInputs, LLMController
from src.utils.static import save_images, save_file, loads_static
from src.utils.client import get_client, Client
from src.utils.response import ResponseData
from src.utils.image_utils import visualize_image_with_mask


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
        self.app.post("/api/uploads")(self.uploads)
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

        # Format results image
        _overlays = visualize_image_with_mask(_images, _results)

        # Save results
        _mask_path = save_images("crackseg_masks", _results)
        _overlay_path = save_images("crackseg_results", _overlays)

        # Update client data
        _response = {
            "masks": _mask_path,
            "results": _overlay_path
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
        _masks = client.data.get("masks", None)

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
