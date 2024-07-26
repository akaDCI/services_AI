from fastapi import FastAPI
from app.routers import image_evaluate

app = FastAPI()

app.include_router(image_evaluate.router, prefix="/api/v1")
