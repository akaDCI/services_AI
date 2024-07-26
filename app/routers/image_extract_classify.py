from fastapi import APIRouter
from app.schemas import *
import json
import base64
import vertexai
import os
from vertexai.generative_models import GenerativeModel, Part, FinishReason
import vertexai.preview.generative_models as generative_models

router = APIRouter()
cwd = os.getcwd()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]= os.path.join(cwd, "app/configs/gemini_auth.json")

def generate(base64image):
    vertexai.init(project="gemini2024-418809", location="us-central1")
    model = GenerativeModel(
        "gemini-1.0-pro-vision-001",
        # system_instruction=[textsi_1]
    )
    image1 = Part.from_data(
        mime_type="image/jpeg",
        data=base64.b64decode(base64image))
    text1 = """Hãy cung cấp thông tin về vật phẩm theo định dạng json mẫu sau.:
{   name: str
    dynasty: str
    category: str
    age: str
    material: str
    description: str
    decoration: str
    crafting_technique: str }"""

    textsi_1 = """Bạn là một nhà thẩm định cổ vật của Việt Nam, nhiệm vụ của bạn là đưa ra thôn tin chính xác về cổ vật bao gồm niên đại, triều, thời gian và lý do bạn đưa ra thẩm định đó. Hãy trở nên hữu dụng, có ích và làm hết khả năng. Chỉ đưa ra thông tin và không yêu cầu cung cấp thông tin gì thêm."""

    generation_config = {
        "max_output_tokens": 2048,
        "temperature": 0.4,
        "top_p": 0.4,
        "top_k": 32,
    }
    responses = model.generate_content(
        [image1, text1],
        generation_config=generation_config,
        safety_settings=safety_settings,
        #   stream=True,
    )

    return responses


safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}



@router.post('/object_attrib', response_model=ObjectAttribute)
def get_object_attrib(image: UploadImage, sample="false"):
    if sample == "true":
        test_data = r"""{
"name": "Mũ miện vua quan thời Nguyễn",
"dynasty": "Nguyễn",
"category": "Mũ",
"age": "Triều Nguyễn",
"material": "Vàng, bạc, ngọc trai, nhung",
"description": "Mũ miện vua quan thời Nguyễn là một trong những loại mũ miện đẹp nhất và tinh xảo nhất trong lịch sử Việt Nam. Mũ miện này được làm bằng vàng, bạc, ngọc trai và nhung, với những họa tiết rồng phượng tinh xảo.",
"decoration": "Rồng, phượng",
"crafting_technique": "Thêu, chạm trổ"
}"""
        # print(test_data)
        return json.loads(test_data)
    else:
        data = generate(base64image=image.base64Image)
        return json.loads(data.text)
