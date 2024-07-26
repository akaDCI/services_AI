from pydantic import BaseModel


class UploadImage(BaseModel):
    base64Image: str

class ImageEvaluateResponse(BaseModel):
    name: str
    dynasty: str
    age: str
    material: str
    description: str
    
class ObjectAttribute(BaseModel):
    name: str
    dynasty: str
    category: str
    age: str
    material: str
    description: str
    decoration: str
    crafting_technique: str
    