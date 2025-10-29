from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch
import pandas as pd
import pickle
import torch.nn.functional as F
import os
import numpy as np
from itertools import compress
import fitz  # this is pymupdf
from itertools import takewhile
import re
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from itertools import chain