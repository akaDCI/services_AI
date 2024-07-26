FROM python:3.10.13-slim-bookworm

# Set environment variables
ENV CLOUD_HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

RUN apt-get update && apt-get install libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 -y

# Setup new user named user with UID 1000
RUN useradd -m -u 1000 user

# Define working directory
WORKDIR $CLOUD_HOME/app

# Switch to the "user" user
USER user

COPY --chown=user:user . .

# RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r $CLOUD_HOME/app/requirements.txt

# Expose the port
EXPOSE 7860/tcp

# CMD ["python", "main.py"]
# Run the application
ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
# ENTRYPOINT ["python", "main.py"]