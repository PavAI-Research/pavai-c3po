from pavai.setup import config 
from pavai.setup import logutil
logger = logutil.logging.getLogger(__name__)
import time
import torch 
from diffusers import StableDiffusionXLPipeline
from abc import ABC, abstractmethod
import gc
import traceback
import pavai.shared.system_types as system_types

__author__ = "mychen76@gmail.com"
__copyright__ = "Copyright 2024"
__version__ = "0.0.3"

DEFAULT_TEXT_TO_IMAGE_MODEL="segmind/SSD-1B"

class AbstractImageGenerationClass(ABC):
    """
    The Abstract Class defines a template method that contains a skeleton of
    Image Generation algorithm, composed of calls to (usually) abstract primitive
    operations.
    Concrete subclasses should implement these operations, but leave the
    template method itself intact.
    """
    def __init__(self, model_id_or_path:str, use_device:str="cpu", 
                 use_torch_type:str="float16",
                 use_download_root:str="resources/models/image-generation",
                 use_local_model_only:bool=False) -> None:
        super().__init__()
        self._model_id_or_path = model_id_or_path
        self._use_device = use_device
        self._use_torch_type = use_torch_type 
        self._use_task = "text-to-image"                                    
        self._model = None
        self._tokenizer = None                      
        self._processor = None                                
        self._pipeline = None 
        self._download_root = use_download_root
        self._local_model_only = use_local_model_only
        self._switch_to_cpu=False

    @property
    def model_id_or_path(self):
        return self._model_id_or_path
    
    @model_id_or_path.setter
    def model_id_or_path(self, new_name):
        self._model_id_or_path = new_name

    @property
    def use_device(self):
        return self._use_device
    
    @use_device.setter
    def use_device(self, new_name):
        self._use_device = new_name

    @property
    def use_local_model_only(self):
        return self._local_model_only
    
    @use_local_model_only.setter
    def use_local_model_only(self, new_name):
        self._local_model_only = new_name

    @property
    def use_download_root(self):
        return self._download_root
    
    @use_download_root.setter
    def use_download_root(self, new_name):
        self._download_root = new_name

    @property
    def use_torchtype(self):
        return self._use_torchtype
    
    @use_torchtype.setter
    def use_torchtype(self, new_name):
        self._use_torchtype = new_name

    @property
    def use_task(self):
        return self._use_task
    
    @use_task.setter
    def use_task(self, new_name):
        self._use_task = new_name

    def generate_image(self, user_prompt:str, 
                  neg_prompt:str="ugly, blurry, poor quality", 
                  output_filename="new_text_to_image_1.png",
                  storage_path:str="workspace/text-to-image/")->str:
        """
        The template method defines the skeleton of text_to_image algorithm.
        """
        print(f"use_device: {self.use_device}")
        try:
            self.load_model()
            self.create_pipeline()
            self.prepare_input()
            self.prepare_output()
            self.hook1()
        except Exception as e:
            print("Exception ocurred ",e.args)
            print(traceback.format_exc())
            if "CUDA out of memory" in str(e.args):
                ## attempt generate using CPU only ONCE 
                self._switch_to_cpu=True
                self.load_model()
                self.create_pipeline()
                self.prepare_input()
                self.prepare_output()
                self.hook1()
        return None

    def free_resources(self):
        del self._pipeline
        self._pipeline = None
        torch.cuda.empty_cache()        
        logger.debug("free resources:",gc.collect())            

    # These operations already have implementations.
    @abstractmethod
    def load_model(self) -> None:
        pass

    @abstractmethod
    def create_pipeline(self) -> None:
        pass

    @abstractmethod
    def prepare_input(self) -> None:
        pass

    @abstractmethod
    def prepare_output(self) -> None:
        pass

    # These are "hooks." Subclasses may override them, but it's not mandatory
    def hook1(self) -> None:
        pass

class StableDiffusionXL(AbstractImageGenerationClass):
    """
    FasterTranscriber Class override only required class operations.
    """
    def load_model(self) -> None:
        logger.debug(f"StableDiffusionXL: load_model: None")                                                

    def load_tokenizer(self) -> None:
        logger.debug(f"StableDiffusionXL: load_tokenizer: None")                                                

    def create_pipeline(self) -> None:
        logger.debug(f"StableDiffusionXL: create_pipeline {self.model_id_or_path}")                                        
        if self._pipeline is None:
            self.use_device = "cuda" if torch.cuda.is_available() else "cpu"
            if self._switch_to_cpu:
                self._use_device="cpu"
            logger.debug(f"use device: {self.use_device}")    
            if self.use_device=="cuda":        
                self._pipeline = StableDiffusionXLPipeline.from_pretrained(self.model_id_or_path, 
                                                     torch_dtype=torch.float16, 
                                                     use_safetensors=True, variant="fp16")
            else:
                self._pipeline = StableDiffusionXLPipeline.from_pretrained(self.model_id_or_path, 
                                                     torch_dtype=torch.float32, 
                                                     use_safetensors=True, variant="fp32")
            self._pipeline.to(self.use_device)
            print(f"StableDiffusionXL: create_pipeline on device {self.use_device}")                                                    

    def prepare_input(self) -> None:
        logger.debug(f"StableDiffusionXL: prepare_input: None")                                                

    def prepare_output(self) -> None:
        logger.debug(f"StableDiffusionXL: prepare_output: None")                                                

    def generate_image(self, user_prompt:str, 
                  neg_prompt:str="ugly, blurry, poor quality", 
                  output_filename="text_to_image_1.png",
                  storage_path:str="workspace/text-to-image/")->str:
        """
        The template method defines the skeleton of generate_image algorithm.
        """
        logger.debug(f"generate_image: {self.use_device}")                                                
        t0 = time.perf_counter()                
        try:        
            self.load_model()
            self.load_tokenizer()
            self.create_pipeline()
            self.prepare_input()
            if user_prompt is None or len(user_prompt)==0:
                return None
            # load on-demand to reduce memory usage at cost of lower performance
            output_filename = storage_path+output_filename
            newImg1 = self._pipeline(user_prompt, negative_prompt=neg_prompt).images[0]
            newImg1.save(output_filename)        
            took_in_seconds = time.perf_counter()-t0        
            status_msg=f"generate_image completed took {took_in_seconds:.2f} seconds"    
            logger.info(status_msg)        
            logger.debug(status_msg)                                                
            self.prepare_output()
            self.hook1()
        except Exception as e:
            print("Exception ocurred ",e.args)
            print(traceback.format_exc())
            if "CUDA out of memory" in str(e.args):
                ## attempt generate using CPU only ONCE 
                self._switch_to_cpu=True
                self.load_model()
                self.create_pipeline()
                self.prepare_input()
                if user_prompt is None or len(user_prompt)==0:
                    return None
                # load on-demand to reduce memory usage at cost of lower performance
                output_filename = storage_path+output_filename
                newImg1 = self._pipeline(user_prompt, negative_prompt=neg_prompt).images[0]
                newImg1.save(output_filename)        
                took_in_seconds = time.perf_counter()-t0        
                status_msg=f"generate_image completed took {took_in_seconds:.2f} seconds"    
                logger.info(status_msg)        
                logger.debug(status_msg)                                                                
                self.prepare_output()
                self.hook1()   
        finally:
                self.free_resources()     
        return output_filename

class StableDiffusionAPI(AbstractImageGenerationClass):
    def generate_image(self, user_prompt:str, 
                    neg_prompt:str="ugly, blurry, poor quality", 
                    output_filename="new_text_to_image_1.png",
                    storage_path:str="workspace/text-to-image/")->str:
        """
        The template method defines the skeleton of text_to_image algorithm.
        """
        import requests
        session = requests.Session()
        authentication = {"USER":"", "PASSWORD":""}
        payload = {"prompt":"some query",
                   "num_inference_steps":12,
                   "guidance_scale":1,
                   "seed":123}
        local_file = "new_image.png"
        # This is a dummy URL. You can replace this with the actual URL
        URL = "http://localhost:8885/api/generate"

        # This is a POST request
        with session.post(URL, stream=True, data=payload, 
                        auth=(authentication["USER"], 
                                authentication["PASSWORD"]), verify=False) as r:
            r.raise_for_status()
            with open(local_file, 'wb') as f:
                for chunk in r.iter_content(chunk_size=128):
                    f.write(chunk)
        return None


    def load_model(self) -> None:
        pass

    def create_pipeline(self) -> None:
        pass

    def prepare_input(self) -> None:
        pass

    def prepare_output(self) -> None:
        pass

# disable cache
#@functools.lru_cache
def image_generation_client(abstract_image_generator: AbstractImageGenerationClass, 
                       user_prompt:str, neg_prompt:str="ugly, blurry, poor quality", 
                       output_filename="new_text_to_image_1.png",
                       storage_path:str="workspace/text-to-image")->str:
    result_file = abstract_image_generator.generate_image(user_prompt=user_prompt, 
                                                     neg_prompt=neg_prompt, 
                                                     output_filename=output_filename,
                                                     storage_path=storage_path)
    return result_file


# #@functools.lru_cache
# def load_text_to_image_model(model_id:str="segmind/SSD-1B", torch_dtype=torch.float16):
#     pipe = StableDiffusionXLPipeline.from_pretrained(model_id, 
#                                                      torch_dtype=torch_dtype, 
#                                                      use_safetensors=True, variant="fp16")
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     pipe.to(device)
#     return pipe

# def text_to_image(user_prompt:str, 
#                   neg_prompt:str="ugly, blurry, poor quality", 
#                   output_filename="new_text_to_image_1.png",
#                   storage_path:str="workspace/text-to-image")->str:
#     if user_prompt is None or len(user_prompt)==0:
#         return None
#     # load on-demand to reduce memory usage at cost of lower performance
#     stablediffusion_model = StableDiffusionXL(model_id_or_path=DEFAULT_TEXT_TO_IMAGE_MODEL)
#     output_filename = image_generation_client(stablediffusion_model,user_prompt=user_prompt,neg_prompt=neg_prompt,
#                                               output_filename=output_filename,storage_path=storage_path)
#     return output_filename
#  test -1 
#  year of dragon
#  inspiration year of dragon 
#  inspiration year of dragon in cartoon style
#  inspiration year of dragon in cartoon style with a happy face
# test -2
# 
