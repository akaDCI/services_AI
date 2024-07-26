import logging
from logging.handlers import RotatingFileHandler
from elasticsearch import Elasticsearch
import threading

class LogSystem:
   _instance = None
   _lock = threading.Lock()

   def __new__(cls, *args, **kwargs):
      with cls._lock:
         if cls._instance is None:
               cls._instance = super(LogSystem, cls).__new__(cls, *args, **kwargs)
               cls._instance._initialize(*args, **kwargs)
         return cls._instance

   def _initialize(self, es_host='localhost', es_port=9200, log_file='logfile.log', es_scheme='http'):
      path = [os.getenv('ELASTICSEARCH_URL', 'http://localhost:9200')]
      self.es = Elasticsearch(path)
      
      # check if Elasticsearch is running
      if not self.es.ping():
            raise ValueError("Elasticsearch is not running")
      else:
            print("Elasticsearch is running")
      
      self.logger = logging.getLogger('ElasticsearchLogger')
      self.logger.setLevel(logging.INFO)
      
      handler = RotatingFileHandler(log_file, maxBytes=2000, backupCount=5)
      formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
      handler.setFormatter(formatter)
      self.logger.addHandler(handler)
        
   def log_and_store_message(self, message, user_name, object_name, timestamp, age):
      self.logger.info(message)
      
      log_entry = {
         "user_name": user_name,
         "object_name": object_name,
         "age": age,
         "message": message,
         "timestamp": timestamp
      }
      
      try:
            res = self.es.index(index="log_index", document=log_entry)
            print(f"Stored message in Elasticsearch: {res['result']}")
      except Exception as e:
            self.logger.error(f"Failed to store message in Elasticsearch: {e}")
            print(f"Failed to store message in Elasticsearch: {e}")

   def search_logs(self, query):
      res = self.es.search(index="log_index", body={"query": {"match": {"message": query}}})
      return res
   
   def get_all_logs(self):
      res = self.es.search(index="log_index", body={"query": {"match_all": {}}}, size=100)
      return res
   
   def get_logs_by_day(self, start_date, end_date):
      
      # Định nghĩa ngày bắt đầu và ngày kết thúc
      # start_date = "2024-07-26T00:00:00"
      # end_date = "2024-07-26T23:59:59"

      # Tạo query để lấy dữ liệu theo ngày
      query = {
         "query": {
            "range": {
                  "timestamp": {
                     "gte": start_date,
                     "lte": end_date,
                     "format": "strict_date_optional_time"
                  }
            }
         }
      }

      # Thực hiện truy vấn
      res = self.es.search(index="log_index", body=query, size=100)
      return res