from typing import Annotated, Any

from pydantic import BaseModel

class GenericResponseModel(BaseModel):
    """Generic response model for all responses"""
    api_id: Annotated[str, "API Id"] = None
    error: Annotated[str, "Error Title"] = None
    message: Annotated[str, "Message"] = None
    data: Any = None
    status_code: Annotated[int, "Status Code"] = None
