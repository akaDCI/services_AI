from fastapi import FastAPI
from app.routers import image_evaluate, image_extract_classify
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Response

app = FastAPI(root_path="/api/v1")

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(image_evaluate.router)
app.include_router(image_extract_classify.router)
