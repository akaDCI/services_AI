from pydantic import BaseModel
from typing import List

class ClusterItem(BaseModel):
    gender: str
    age: str
    career: str
    interest: str

class ClusterItemList(BaseModel):
    items: List[ClusterItem]