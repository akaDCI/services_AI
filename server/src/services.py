from dataclasses import dataclass, field
import orjson
from fastapi import FastAPI, Request, Response

@dataclass
class Services:
          """API Services"""
          app: "FastAPI" = field(default_factory=FastAPI)
          request: Request = field(default=None)
          response: Response = field(default=None)

          # define router here
          def __post_init__(self):
              """Post init"""
              self.app.get("/")(self.hello_world)

          
          # write router function here
          async def hello_world(self, request: Request, response: Response):
                    body = await request.body()
                    item = orjson.loads(body)
                    return {"message": "Hello World"}
          
          @property
          def __call__(self):
              return self.app