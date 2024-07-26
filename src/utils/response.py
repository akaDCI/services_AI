from pydantic import BaseModel
from fastapi.responses import ORJSONResponse


class ResponseModel(BaseModel):
    msg: str
    data: dict


def ResponseData(data: dict, status: int = 200, msg: str = "Success") -> ORJSONResponse:
    return ORJSONResponse(
        content=ResponseModel(
            msg=msg,
            data=data
        ).model_dump(),
        status_code=status
    )
