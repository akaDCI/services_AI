from pydantic import BaseModel


class UploadImage(BaseModel):
    base64Image: str

class ImageEvaluateResponse(BaseModel):
    name: str
    dynasty: str
    age: str
    material: str
    description: str