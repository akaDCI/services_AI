import http
import json

import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from contextlib import asynccontextmanager

from controllers import status, cluster_vis_pred_controller
from models.base import GenericResponseModel
# from logs import logger
from utils.helper import build_api_response
from scripts.context_manager import context_log_meta
from utils.exceptions import AppException

# register startup and shutdown using lifespan Events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup event
    # logger.info("Startup Event Triggered")
    print("Startup Event Triggered")
    yield
    # shutdown event
    # logger.info("Shutdown Event Triggered")
    print("Shutdown Event Triggered")

app = FastAPI(lifespan=lifespan)
app.title = "akaDCI API Services"
app.version = "0.0.1"

# register routers here and add dependency on authenticate_token if token based authentication is required
app.include_router(status.status_router)
app.include_router(cluster_vis_pred_controller.cluster_vis_pred)

# register exception handlers here
@app.exception_handler(ValidationError)
async def pydantic_validation_exception_handler(request: Request, exc):
    # logger.error(extra=context_log_meta.get(), msg=f"data validation failed {exc.errors()}")
    return build_api_response(GenericResponseModel(status_code=http.HTTPStatus.BAD_REQUEST,
                                                   error=f"Data validation failed: {json.loads(str(exc))}"))

@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc):
    # logger.error(extra=context_log_meta.get(), msg=f"application exception occurred error: {json.loads(str(exc))}")
    return build_api_response(GenericResponseModel(status_code=exc.status_code, error=f"Application exception occurred error: {json.loads(str(exc))}"))

# not_found_exception_handler
@app.exception_handler(404)
async def not_found_exception_handler(request: Request, exc: HTTPException):
    # logger.error(extra=context_log_meta.get(), msg=f"Resource not found: {exc}")
    return build_api_response(GenericResponseModel(status_code=exc.status_code, error="Resource not found"))

# request_validation_exception_handler
@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request: Request, exc):
    # logger.error(extra=context_log_meta.get(), msg=f"Request Validation Failed: {exc}")
    return build_api_response(GenericResponseModel(status_code=http.HTTPStatus.BAD_REQUEST,
                                                   error=f"Request Validation Failed: {json.loads(str(exc))}"))

# Create a GET method that responds with HTML code
@app.get('/', tags = ['home'])
def message():
    return HTMLResponse('<h1>Welcome to akaDCI API Services</h1>')

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=True)