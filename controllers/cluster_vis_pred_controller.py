import http
import json
from fastapi import APIRouter, HTTPException, Body

from models.base import GenericResponseModel
from models.cluster_schemas import ClusterItem, ClusterItemList
from services.cluster_vis_pred_service import ChallengeClusterService

cluster_vis_pred = APIRouter(prefix="/api")

@cluster_vis_pred.post("/vis", status_code=http.HTTPStatus.OK, response_model=GenericResponseModel)
async def visualize(num_clusters: int, json_data: ClusterItemList):
    challenge_service = ChallengeClusterService()
    # Load data
    with open('data/cluster_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # json_data_dict = [item.__dict__ for item in json_data.items]
    # # Convert the dictionary to a JSON string
    # json_string = json.dumps(json_data_dict)
    # # Load the JSON string into a Python object
    # data = json.loads(json_string)
    challenge_service.visuallize(data, num_clusters)
    return GenericResponseModel(status_code=http.HTTPStatus.OK, message="Successfully", data=None)