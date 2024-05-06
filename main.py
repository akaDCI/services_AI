import logging
import uvicorn
from src.services import Services

class AI_server:
          
          def __init__(self):
                    self.api = Services()
          
          def __call__(self):
                  return self.api.app
        
        
if __name__ == "__main__":
          server = AI_server()
          logging.log(logging.INFO, "Starting server")
          uvicorn.run(server, host="0.0.0.0", port=8000)