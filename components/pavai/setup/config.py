from __future__ import annotations
import gc
import traceback
import os
import sys
from dotenv import dotenv_values
system_config = {
    **dotenv_values("env.shared"),  # load shared development variables
    **dotenv_values("env.secret"),  # load sensitive variables
    **os.environ,  # override loaded values with environment variables
}

import torch
from transformers.utils import is_flash_attn_2_available

system_cpus_count = os.cpu_count()
#torch.manual_seed(10)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.set_num_threads(int(system_cpus_count/2))

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

print("system cpu:",system_cpus_count)
print("system device:",device)
print("system compute_type:",compute_type)
print("system torch_type:",use_torch_dtype)
print("system use_flash_attention_2:",use_flash_attention_2)

# import random
# random.seed(0)
# import numpy as np
# np.random.seed(0)
#import nltk
#nltk.download('punkt')

