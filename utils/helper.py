import http
import uuid
import re

from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from scripts.context_manager import context_api_id, context_log_meta
# from logs import logger
from models.base import GenericResponseModel


def build_api_response(generic_response: GenericResponseModel) -> JSONResponse:
    try:
        if not generic_response.api_id:
            generic_response.api_id = context_api_id.get() if context_api_id.get() else str(uuid.uuid4())
        if not generic_response.status_code:
            generic_response.status_code = http.HTTPStatus.OK if not generic_response.error \
                else http.HTTPStatus.UNPROCESSABLE_ENTITY
        response_json = jsonable_encoder(generic_response)
        res = JSONResponse(status_code=generic_response.status_code, content=response_json)
        # logger.info(extra=context_log_meta.get(), msg="build_api_response: Generated Response with status_code:"+ f"{generic_response.status_code}")
        return res
    except Exception as e:
        # logger.error(extra=context_log_meta.get(), msg=f"exception in build_api_response error : {e}")
        return JSONResponse(status_code=generic_response.status_code, content=generic_response.error)

 
# validating an Email input
def check_email_input(email):
    regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b'
    # pass the regular expression
    # and the string into the fullmatch() method
    if(re.fullmatch(regex, email)):
        return True
    else:
        return False