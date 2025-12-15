eval "$(conda shell.bash hook)"
conda activate llama_env

nohup python -m llama_cpp.server \
--model /mnt/c/Users/user/.lmstudio/models/lmstudio-community/granite-4.0-micro-GGUF/granite-4.0-micro.gguf   \
--port 8081 > logs/ibm_granite.log  2>&1 &
