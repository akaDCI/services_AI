# Sử dụng hình ảnh chính thức của Python làm hình ảnh cơ bản
FROM python:3.9-slim

# Cập nhật pip
RUN pip install --upgrade pip

# Thiết lập thư mục làm việc
WORKDIR /app

# Sao chép tệp requirements
COPY requirements.txt .

# Cài đặt các thư viện phụ thuộc
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép mã ứng dụng FastAPI
COPY . .

# Mở cổng FastAPI
EXPOSE 8000

# Lệnh để chạy ứng dụng FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]