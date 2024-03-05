from __future__ import annotations
import os
import sys
import gc
import traceback
import torch
from rich.console import Console 
from transformers.utils import is_flash_attn_2_available
from dotenv import dotenv_values
system_config = {
    **dotenv_values("env.shared"),  # load shared development variables
    **dotenv_values("env.secret"),  # load sensitive variables
    **os.environ,  # override loaded values with environment variables
}
system_cpus_count = os.cpu_count()
#torch.manual_seed(10)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.set_num_threads(int(system_cpus_count/2))

## disable telemetry
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
os.environ['HF_TOKEN']='1'

## get hardward info
omp_cpus_usage=str(int(os.cpu_count()/2))
os.environ["OMP_NUM_THREADS"] = omp_cpus_usage
device = "cuda" if torch.cuda.is_available() else "cpu"
use_device = "cuda:0" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if torch.cuda.is_available() else "int8"
#compute_type = torch.float16 if torch.cuda.is_available() else torch.float32
use_torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
use_flash_attention_2 = is_flash_attn_2_available()

INT16_MAX_ABS_VALUE = 32768.0
DEFAULT_MAX_MIC_RECORD_LENGTH_IN_SECONDS = 30*60*60  # 30 miniutes

console  = Console()
console.print("-----ENVIRONMENT------------------")
console.print("system cpu:",system_cpus_count)
console.print("system device:",device)
console.print("system compute_type:",compute_type)
console.print("system torch_type:",use_torch_dtype)
console.print("system use_flash_attention_2:",use_flash_attention_2)
console.print("torch version: ",torch.__version__)
console.print("----------------------------------")

# import random
# random.seed(0)
# import numpy as np
# np.random.seed(0)
#import nltk
#nltk.download('punkt')
