FROM nvidia/cuda:12.1.0-base-ubuntu22.04 

RUN apt-get update -y \
    && apt-get install -y python3-pip

RUN ldconfig /usr/local/cuda-12.1/compat/

# Install Python dependencies
COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install --upgrade -r /requirements.txt

# Install vLLM (switching back to pip installs since issues that required building fork are fixed and space optimization is not as important since caching) and FlashInfer 
RUN python3 -m pip install vllm==0.12.0 && \
    python3 -m pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.3

# Setup for Option 2: Building the Image with the Model included
ARG BASE_PATH="/runpod-volume"
ARG MODEL_NAME=volvetech/Kimi-K2-Thinking
ARG TRUST_REMOTE_CODE=true
ARG TENSOR_PARALLEL_SIZE=8
ARG MAX_MODEL_LEN=262144
ARG MAX_NUM_BATCHED_TOKENS=32768
ARG TOOL_CALL_PARSER=kimi_k2
ARG REASONING_PARSER=kimi_k2

ENV MODEL_NAME=$MODEL_NAME \
    TRUST_REMOTE_CODE=$TRUST_REMOTE_CODE \
    TENSOR_PARALLEL_SIZE=$TENSOR_PARALLEL_SIZE \
    MAX_MODEL_LEN=$MAX_MODEL_LEN \
    MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS \
    TOOL_CALL_PARSER=$TOOL_CALL_PARSER \
    REASONING_PARSER=$REASONING_PARSER \
    BASE_PATH=$BASE_PATH \
    HF_DATASETS_CACHE="${BASE_PATH}/huggingface-cache/datasets" \
    HUGGINGFACE_HUB_CACHE="${BASE_PATH}/huggingface-cache/hub" \
    HF_HOME="${BASE_PATH}/huggingface-cache/hub" \
    HF_HUB_ENABLE_HF_TRANSFER=0 

ENV PYTHONPATH="/:/vllm-workspace"


COPY src /src

# Start the handler
CMD ["python3", "/src/handler.py"]
