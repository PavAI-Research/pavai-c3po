
from src.shared.hf_utils import hf_model_snapshot_download, hf_model_file_download,hf_multimodal_files_download


def download_repository_snapshots(snapshot_file:str="hf_models_snapshot.json",cache_dir:str="./models"):
    model_name_or_path="BAAI/bge-small-en-v1.5"
    notes="embedding model"
    hf_model_snapshot_download(model_name_or_path=model_name_or_path,snapshot_file=snapshot_file,cache_dir=cache_dir,append_file=False,notes=notes)
    
    model_name_or_path="scroobiustrip/topic-model-v3"
    notes="text topic modelling"
    hf_model_snapshot_download(model_name_or_path=model_name_or_path,snapshot_file=snapshot_file,cache_dir=cache_dir,append_file=True,notes=notes)

    model_name_or_path="cardiffnlp/tweet-topic-21-multi"
    notes="text topic classification"
    hf_model_snapshot_download(model_name_or_path=model_name_or_path,snapshot_file=snapshot_file,cache_dir=cache_dir,append_file=True,notes=notes)

def download_repository_domains(model_config_file:str="hf_domain_models.json",cache_dir:str="./models"):
    ## default models
    model_name_or_path="abetlen/functionary-7b-v1-GGUF"
    model_file="functionary-7b-v1.Q4_K_S.gguf"

    model_alias="functionary"
    chat_format="llama-2"
    notes="model support function call model"

    hf_model_file_download(model_name_or_path=model_name_or_path,
                        model_file=model_file,
                            model_alias=model_alias,
                            chat_format=chat_format,
                            snapshot_file=model_config_file,
                        cache_dir=cache_dir,
                        append_file=False,notes=notes)



    model_name_or_path="TheBloke/LlamaGuard-7B-GGUF"
    model_file="llamaguard-7b.Q2_K.gguf"

    model_alias="llamaguard"
    chat_format="llama-2"
    notes="for content safety guard"

    hf_model_file_download(model_name_or_path=model_name_or_path,
                        model_file=model_file,
                            model_alias=model_alias,
                            chat_format=chat_format,
                            snapshot_file=model_config_file,
                        cache_dir=cache_dir,
                        append_file=True,notes=notes)


    model_name_or_path="TheBloke/zephyr-7B-beta-GGUF"
    model_file="zephyr-7b-beta.Q4_K_M.gguf"

    model_alias="zephyr-7b-beta.Q4"
    chat_format="chatml"
    notes="general model"

    hf_model_file_download(model_name_or_path=model_name_or_path,
                        model_file=model_file,
                            model_alias=model_alias,
                            chat_format=chat_format,
                            snapshot_file=model_config_file,
                        cache_dir=cache_dir,
                        append_file=True,notes=notes)    
    
    ## extented domain models
    model_name_or_path="TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
    model_file="mistral-7b-instruct-v0.2.Q4_K_M.gguf"

    model_alias="mistral-7b-instruct-v0.2.Q4"
    chat_format="llama-2"
    notes="general model"

    hf_model_file_download(model_name_or_path=model_name_or_path,
                        model_file=model_file,
                            model_alias=model_alias,
                            chat_format=chat_format,
                            snapshot_file=model_config_file,
                        cache_dir=cache_dir,
                        append_file=True,notes=notes)


    model_name_or_path="TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF"
    model_file="mixtral-8x7b-instruct-v0.1.Q3_K_M.gguf"

    model_alias="mixtral-8x7b-instruct-v0.1.Q3"
    chat_format="llama-2"
    notes="general model"

    hf_model_file_download(model_name_or_path=model_name_or_path,
                        model_file=model_file,
                            model_alias=model_alias,
                            chat_format=chat_format,
                            snapshot_file=model_config_file,
                        cache_dir=cache_dir,
                        append_file=True,notes=notes)


    model_name_or_path="TheBloke/medicine-LLM-13B-GGUF"
    model_file="medicine-llm-13b.Q3_K_M.gguf"

    model_alias="medicine"
    chat_format="llama-2"
    notes="specialized medicine model"

    hf_model_file_download(model_name_or_path=model_name_or_path,
                        model_file=model_file,
                            model_alias=model_alias,
                            chat_format=chat_format,
                            snapshot_file=model_config_file,
                        cache_dir=cache_dir,
                        append_file=True,notes=notes)


    model_name_or_path="TheBloke/law-LLM-13B-GGUF"
    model_file="law-llm-13b.Q3_K_M.gguf"

    model_alias="law"
    chat_format="llama-2"
    notes="specialized law model"

    hf_model_file_download(model_name_or_path=model_name_or_path,
                        model_file=model_file,
                            model_alias=model_alias,
                            chat_format=chat_format,
                            snapshot_file=model_config_file,
                        cache_dir=cache_dir,
                        append_file=True,notes=notes)


    model_name_or_path="TheBloke/finance-LLM-GGUF"
    model_file="finance-llm.Q4_K_M.gguf"

    model_alias="finance"
    chat_format="llama-2"
    notes="specialized finance model"

    hf_model_file_download(model_name_or_path=model_name_or_path,
                        model_file=model_file,
                            model_alias=model_alias,
                            chat_format=chat_format,
                            snapshot_file=model_config_file,
                        cache_dir=cache_dir,
                        append_file=True,notes=notes)

    model_alias="llava-v1.5-7b"
    model_name_or_path="mys/ggml_llava-v1.5-7b"
    model_file="ggml-model-q5_k.gguf"
    cli_project_file="mmproj-model-f16.gguf"
    chat_format="llava-1-5"
    notes="model support function call model"

    hf_multimodal_files_download(model_name_or_path=model_name_or_path,
                        model_file=model_file,
                        model_alias=model_alias,
                        clip_model_path=cli_project_file,
                        chat_format=chat_format,
                        snapshot_file=model_config_file,
                        cache_dir=cache_dir,
                        append_file=True,
                        notes=notes)

    model_alias="bakllava-1"
    model_name_or_path="mys/ggml_bakllava-1"
    model_file="ggml-model-q5_k.gguf"
    cli_project_file="mmproj-model-f16.gguf"
    chat_format="llava-1-5"
    notes="model support function call model"

    hf_multimodal_files_download(model_name_or_path=model_name_or_path,
                        model_file=model_file,
                        model_alias=model_alias,
                        clip_model_path=cli_project_file,
                        chat_format=chat_format,
                        snapshot_file=model_config_file,
                        cache_dir=cache_dir,
                        append_file=True,
                        notes=notes)


if __name__=="__main__":
    download_repository_snapshots()
    download_repository_domains()
    