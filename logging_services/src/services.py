import orjson
from dataclasses import dataclass, field
from fastapi import FastAPI, Request, Response
from fastapi.responses import RedirectResponse
from .logging_services.logging_system import LogSystem
from datetime import datetime
from .logging_services.gpt_model import OpenAIClient
from .logging_services.load_systhesis import read_and_load_data

AZURE_ENDPOINT = ""
API_KEY = ""


@dataclass
class Services:
    """API Services"""
    app: "FastAPI" = field(default_factory=FastAPI)
    request: Request = field(default=None)
    response: Response = field(default=None)

    # define router here
    def __post_init__(self):
        """Post init"""
        self.log_system = LogSystem()

        # Register routes
        self.app.get("/")(self.main)
        self.app.post("/api/test_logging")(self.add_logging)
        self.app.get("/api/get_all_logs")(self.get_all_logs)
        self.app.get("/api/get_logs_by_day")(self.get_logs_by_day)
        self.app.post("/api/statis_hourly_distribution")(self.statis_hourly_distribution)
        self.app.post("/api/statis_daily_distribution")(self.statis_daily_distribution)
        self.app.post("/api/statis_object_distribution")(self.statis_object_distribution)
        self.app.post("/api/statis_age_distribution")(self.statis_age_distribution)
        self.app.post("/api/load_fake_data")(self.load_fake_data)

    async def main(self, request: Request, response: Response):
        """
        Redirect to the Swagger documents
        """
        return RedirectResponse("/docs")
    
    async def load_fake_data(self):
        """
        Load fake data
        """
        # read file data.json
        read_and_load_data()
        return {"message": "Loaded fake"}
    
    async def statis_hourly_distribution(self, request: Request):
        """
        Statis logging
        """
        body = await request.body()
        item = orjson.loads(body)
        start_date = item["start_date"]
        end_date = item["end_date"]
        data = []
        log_by_day = self.log_system.get_logs_by_day(start_date, end_date)
        for hit in log_by_day['hits']['hits']:
            data.append(hit["_source"])
        # viz = Visualization()
        # hourly_path = viz.plot_hourly_distribution(data, 'timestamp')
        task = "hourly distribution data report"
        gpt_model = OpenAIClient(azure_endpoint=AZURE_ENDPOINT, api_key=API_KEY)
        knowledge = str(data)
        bot_answer = gpt_model.call_openai(data_info=knowledge, task=task)
        return {"path_storage": data,
                "bot_answer": bot_answer}
        
    async def statis_daily_distribution(self, request: Request):
        """
        Statis logging
        """
        body = await request.body()
        item = orjson.loads(body)
        start_date = item["start_date"]
        end_date = item["end_date"]
        data = []
        log_by_day = self.log_system.get_logs_by_day(start_date, end_date)
        for hit in log_by_day['hits']['hits']:
            data.append(hit["_source"])
        # viz = Visualization()
        # daily_path = viz.plot_daily_distribution(data, 'timestamp')
        task = "daily distribution data report"
        gpt_model = OpenAIClient(azure_endpoint=AZURE_ENDPOINT, api_key=API_KEY)
        knowledge = str(data)
        bot_answer = gpt_model.call_openai(data_info=knowledge, task=task)
        return {"path_storage": data,
                "bot_answer": bot_answer}
        
    async def statis_object_distribution(self, request: Request):
        """
        Statis logging
        """
        body = await request.body()
        item = orjson.loads(body)
        start_date = item["start_date"]
        end_date = item["end_date"]
        data = []
        log_by_day = self.log_system.get_logs_by_day(start_date, end_date)
        for hit in log_by_day['hits']['hits']:
            data.append(hit["_source"])
        # viz = Visualization()
        # object_path = viz.plot_object_distribution(data, 'object_name')
        task = "object distribution data report"
        gpt_model = OpenAIClient(azure_endpoint=AZURE_ENDPOINT, api_key=API_KEY)
        knowledge = str(data)
        bot_answer = gpt_model.call_openai(data_info=knowledge, task=task)
        return {"path_storage": data,
                "bot_answer": bot_answer}
    
    
    async def statis_age_distribution(self, request: Request):
        """
        Statis logging
        """
        body = await request.body()
        item = orjson.loads(body)
        start_date = item["start_date"]
        end_date = item["end_date"]
        data = []
        log_by_day = self.log_system.get_logs_by_day(start_date, end_date)
        for hit in log_by_day['hits']['hits']:
            data.append(hit["_source"])
        # viz = Visualization()
        # age_path = viz.plot_age_distribution(data, 'age')
        task = "age distribution data report"
        gpt_model = OpenAIClient(azure_endpoint=AZURE_ENDPOINT, api_key=API_KEY)
        knowledge = str(data)
        bot_answer = gpt_model.call_openai(data_info=knowledge, task=task)
        return {"path_storage": data,
                "bot_answer": bot_answer}
        
    async def add_logging(self, request: Request):
        """
        Test logging
        """
        # Example using LogSystem class
        body = await request.body()
        item = orjson.loads(body)
        text = item["text"]
        user_name = item.get("user_name")
        object_name = item.get("object_name")
        age = item.get("age")
        timestamp = datetime.now().isoformat()
        self.log_system.log_and_store_message(text, user_name, object_name, timestamp, age)
        return {"message": "Logged"}
    
    async def get_all_logs(self):
        """
        Get all logs
        """
        data = self.log_system.get_all_logs()
        
        # format data
        logs = []
        for hit in data['hits']['hits']:
            logs.append(hit["_source"])
        return logs
    
    async def get_logs_by_day(self, request: Request):
        """
        Get logs by day
        """
        body = await request.body()
        item = orjson.loads(body)
        start_date = item["start_date"]
        end_date = item["end_date"]
        data = self.log_system.get_logs_by_day(start_date, end_date)
        
        # format data
        logs = []
        for hit in data['hits']['hits']:
            logs.append(hit["_source"])
        return logs

    @property
    def __call__(self):
        return self.app
