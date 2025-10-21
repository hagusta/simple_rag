eval "$(conda shell.bash hook)"
conda activate llama_env

python -m llama_cpp.server \
--model /home/hagusta/.cache/huggingface/hub/models--janhq--TinyLlama-1.1B-Chat-v1.0-GGUF/blobs/74b3c8d38fa6e735e855f3b41f879155d9260e7cf02d3d1ea06a8676d31ac6bd   \
--port 8081 > logs/tiny_lama.log
