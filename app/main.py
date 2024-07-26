from fastapi import FastAPI
from app.routers import image_evaluate, image_extract_classify

app = FastAPI(root_path="/api/v1")

app.include_router(image_evaluate.router)
app.include_router(image_extract_classify.router)
