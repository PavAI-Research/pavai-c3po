
from huggingface_hub import hf_hub_download, snapshot_download

print("manual download model file")

repo_id="TheBloke/zephyr-7B-beta-GGUF"
filename="zephyr-7b-beta.Q4_K_M.gguf"
cache_dir="resources/models/llm"
local_model_path = hf_hub_download(
    repo_id=repo_id, 
    filename=filename, 
    cache_dir=cache_dir)

print("local model path: ", local_model_path)



