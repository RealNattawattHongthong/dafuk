import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from PIL import Image
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')