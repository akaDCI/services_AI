# AkaDCI AI Service

**3D Reconstruction and Virtual Tours: Preserving Historical and Cultural Antiquities with AI**

### Clone this project

```sh
# Clone this project
git clone https://github.com/akaDCI/services_AI.git

# Change directory
cd ./services_AI

# Create virtual environment (Optional)
pip install virtualenv
virtualenv env
env/Scripts/activate

# Install requirements
pip install -r requirements.txt

# Run project
python main.py
```

### Setup for production server

Add these variables to environment key:

- `PORT=7860`
- `HOST=0.0.0.0`

Run Docker locally

```sh
# Build docker image
docker build -t akadci-ai-service .

# Run docker
docker run akadci-ai-service --name akadci-container -p 80:7860 -e HOST=0.0.0.0 -e PORT=7860
```
