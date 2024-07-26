import os
import json
from src.utils.static import StaticDirectory


class Client:
    def __init__(self, client_id: str):
        self._path = os.path.join(os.getcwd(), StaticDirectory, "data.json")
        self.id = client_id
        self.data = self.__fetch_client()

    def __fetch_client(self) -> dict:
        with open(self._path, "r") as f:
            data = json.load(f)
            f.close()
            return data.get(self.id, {})

    def save(self):
        # Read data
        with open(self._path, "r") as f:
            data = json.load(f)
            f.close()

        # Update current client data
        data[self.id] = self.data

        # Save data
        with open(self._path, "w") as f:
            json.dump(data, f)
            f.close()

    def update(self, data: dict):
        self.data.update(data)


def get_client():
    client_id = "test"
    return Client(client_id)
