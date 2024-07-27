from google_cloud_pipeline_components.preview.llm \
import rlhf_pipeline
# Import from KubeFlow pipelines
from kfp import compiler
import google.cloud.aiplatform as aiplatform
import math

RLHF_PIPELINE_PKG_PATH = "rlhf_pipeline.yaml"

compiler.Compiler().compile(
    pipeline_func=rlhf_pipeline,
    package_path=RLHF_PIPELINE_PKG_PATH
)

PREF_DATASET_SIZE = 3000
BATCH_SIZE = 64
REWARD_STEPS_PER_EPOCH = math.ceil(PREF_DATASET_SIZE / BATCH_SIZE)
print(REWARD_STEPS_PER_EPOCH)
REWARD_NUM_EPOCHS = 30
# Calculate number of steps in the reward model training
reward_model_train_steps = REWARD_STEPS_PER_EPOCH * REWARD_NUM_EPOCHS
print(reward_model_train_steps)

# Prompt dataset size
PROMPT_DATASET_SIZE = 2000
# Batch size is fixed at 64
BATCH_SIZE = 64
RL_STEPS_PER_EPOCH = math.ceil(PROMPT_DATASET_SIZE / BATCH_SIZE)
print(RL_STEPS_PER_EPOCH)

RL_NUM_EPOCHS = 10
# Calculate the number of steps in the RL training
reinforcement_learning_train_steps = RL_STEPS_PER_EPOCH * RL_NUM_EPOCHS
print(reinforcement_learning_train_steps)

# Completed values for the dictionary
parameter_values={
        "preference_dataset": \
    "gs://akadci/rlfh/data/summarize_from_feedback_tfds/comparisons/train/*.jsonl",
        "prompt_dataset": \
    "gs://akadci/rlfh/data/historical/train/*.jsonl",
        "eval_dataset": \
    "gs://akadci/rlfh/data/historical/val/*.jsonl",
        "large_model_reference": "llama-2-7b",
        "reward_model_train_steps": 1410,
        "reinforcement_learning_train_steps": 320, # results from the calculations above
        "reward_model_learning_rate_multiplier": 1.0,
        "reinforcement_learning_rate_multiplier": 1.0,
        "kl_coeff": 0.1, 
        "instruction":\
    "Summarize in less than 50 words"}

from utils import authenticate
credentials, PROJECT_ID, STAGING_BUCKET = authenticate()

# RLFH pipeline is available in this region
REGION = "europe-west4"
aiplatform.init(project = PROJECT_ID,
                location = REGION,
                credentials = credentials)

job = aiplatform.PipelineJob(
    display_name="chatbot-historical-rlhf-tuning",
    pipeline_root=STAGING_BUCKET,
    template_path=RLHF_PIPELINE_PKG_PATH,
    parameter_values=parameter_values)

job.run()