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
    text1 = """hãy cho tôi biết thông tin về cổ vật sau của Việt Nam. Hãy trích xuất dưới dạng json theo format ví dụ sau, hãy trả lời chính xác, không được đưa thông tin sai lệch và cụ thể. Nếu không trả lời được hãy để None:
    { \"name\": \"\", 
    \"dynasty\": \"\", 
    \"age\": \"\", 
    \"material\": \"\", 
    \"description\": Lý do đưa ra các thông tin trên }"""

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



@router.post('/eval_image', response_model=ImageEvaluateResponse)
def eval_image(image: UploadImage, sample="false"):
    if sample == "true":
        test_data = r"""{
    "name": "Bội tinh Long tinh",
    "dynasty": "Nhà Nguyễn",
    "age": "1885",
    "material": "Vàng",
    "description": "Là phần thưởng cao quý nhất của triều Nguyễn, dành tặng cho những cá nhân có công lớn với đất nước."
    }"""
        print(test_data)
        return json.loads(test_data)
    else:
        data = generate(base64image=image.base64Image)
        return json.loads(data.text)
