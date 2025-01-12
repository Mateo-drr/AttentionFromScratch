# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 16:40:49 2025

@author: Mateo-drr
"""

import torch
import torch.nn as nn
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace