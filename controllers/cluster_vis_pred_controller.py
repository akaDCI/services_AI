import http
import json
from fastapi import APIRouter, HTTPException, Body

from models.base import GenericResponseModel
from models.cluster_schemas import ClusterItem, ClusterItemList
from services.cluster_vis_pred_service import ChallengeClusterService

cluster_vis_pred = APIRouter(prefix="/api")

@cluster_vis_pred.get("/cluster_vis", status_code=http.HTTPStatus.OK, response_model=GenericResponseModel)
async def visualize(num_clusters: int = 4, synthetic_data: bool = True, json_data: ClusterItemList= None):
    challenge_service = ChallengeClusterService()
    if synthetic_data == False:
        json_data_dict = [item.__dict__ for item in json_data.items]
        # Convert the dictionary to a JSON string
        json_string = json.dumps(json_data_dict)
        # Load the JSON string into a Python object
        data = json.loads(json_string)
    else:
        # Load data
        with open('data/cluster_data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
    json_string = challenge_service.visuallize(data, num_clusters, synthetic_data)
    # save json_string to json file
    with open('data/cluster_vis.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(json_string))
    return GenericResponseModel(status_code=http.HTTPStatus.OK, message="Successfully", data=json_string)

@cluster_vis_pred.get("/cluster_report", status_code=http.HTTPStatus.OK, response_model=GenericResponseModel)
async def get_report(synthetic_data: bool = True, json_data: ClusterItemList= None):
    challenge_service = ChallengeClusterService()
    if synthetic_data == False:
        json_data_dict = [item.__dict__ for item in json_data.items]
        # Convert the dictionary to a JSON string
        json_string = json.dumps(json_data_dict)
        # Load the JSON string into a Python object
        data = json.loads(json_string)
    else:
        # Load data
        with open('data/cluster_data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
    bot_answer = challenge_service.export_report(data, synthetic_data)

    # save bot_answer to txt file
    with open('data/cluster_report.txt', 'w', encoding='utf-8') as f:
        f.write(bot_answer)

    return GenericResponseModel(status_code=http.HTTPStatus.OK, message="Successfully", data=bot_answer)