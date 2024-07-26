import numpy as np
from datetime import datetime, timedelta
from faker import Faker
import random
import json
from .logging_system import LogSystem
# Khởi tạo Faker
faker = Faker()
log_system = LogSystem()

# Định nghĩa các người dùng, cổ vật và các mẫu câu hỏi
user_names = ["tinld", "anhtuan", "bichngoc", "hoangnam", "phuonglinh"]
object_names = [
    "chén uống trà thời Nguyễn", "bình gốm thời Nguyễn", "đĩa sứ thời Nguyễn", 
    "tranh thêu thời Nguyễn", "tượng Phật thời Nguyễn", "áo giáp thời Nguyễn", 
    "mũ thời Nguyễn", "kiếm thời Nguyễn", "trống thời Nguyễn", "sách cổ thời Nguyễn"
]
messages = [
    "Cổ vật này có từ thời kỳ nào?",
    "Vật liệu làm ra cổ vật này là gì?",
    "Kích thước của cổ vật này như thế nào?",
    "Cổ vật này được tìm thấy ở đâu?",
    "Giá trị lịch sử của cổ vật này là gì?"
]

# range from 10 to 30
age = random.randint(10, 30)

# Hàm để tạo timestamp phân phối theo Gaussian
def generate_gaussian_timestamp():
    mean = 12  # Trung bình là giữa ngày (12 giờ trưa)
    std_dev = 3  # Độ lệch chuẩn là 3 giờ
    hours = np.random.normal(mean, std_dev)
    minutes = np.random.normal(30, 15)
    
    if hours < 0:
        hours = 0
    if hours > 23:
        hours = 23
    if minutes < 0:
        minutes = 0
    if minutes > 59:
        minutes = 59

    return datetime.now().replace(hour=int(hours), minute=int(minutes), second=0, microsecond=0)

def systhesis_data():
   data = []
   for _ in range(100):  # Tạo 100 mẫu
      user_name = random.choice(user_names)
      object_name = random.choice(object_names)
      message = random.choice(messages)
      timestamp = generate_gaussian_timestamp().isoformat()
      age = random.randint(10, 30)
      
      data.append({
         "user_name": user_name,
         "object_name": object_name,
         "age": age,
         "message": message,
         "timestamp": timestamp
      })

   # Xuất ra file JSON
   with open('data.json', 'w', encoding='utf-8') as f:
      json.dump(data, f, ensure_ascii=False, indent=4)

   print("Data generated and saved to data.json")
   
def read_and_load_data():
   with open('data.json', 'r', encoding='utf-8') as f:
      data = json.load(f)   
   # Lưu dữ liệu vào Elasticsearch
   for item in data:
      log_system.log_and_store_message(item["message"], item["user_name"], item["object_name"], item["timestamp"], item["age"])
   print("Data loaded from data.json")
   return data

# if __name__ == "__main__":
#    systhesis_data()
#    read_and_load_data()