eval "$(conda shell.bash hook)"
conda activate llama_env

python -m llama_cpp.server \
--model /home/hagusta/.cache/huggingface/hub/models--lmstudio-community--DeepSeek-R1-Distill-Llama-8B-GGUF/blobs/e42c90cb131c7f569364b2cc039ec03b40bc1e648090fc03470f07e3bcde6428   \
--port 8081 > logs/google-gemma.log
