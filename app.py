'''
    Original Paper and repository here : https://github.com/openai/gpt-2
    Pytorch Implementation: https://github.com/graykode/gpt-2-Pytorch
    GPT2 Pytorch Model : https://github.com/huggingface/pytorch-pretrained-BERT
'''
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import random
import numpy as np
import torch

from GPT2.model import (GPT2LMHeadModel)
from GPT2.utils import load_weight
from GPT2.config import GPT2Config
from GPT2.sample import SampleNode
from GPT2.encoder import get_encoder

__all__ = ['model', 'encoder']

state_dict = torch.load('gpt2-pytorch_model.bin', map_location='cpu')
device = torch.device("cpu")

batch_size = 1
encoder = get_encoder()
config = GPT2Config()

model = GPT2LMHeadModel(config)
model = load_weight(model, state_dict)
model.to(device)
model.eval()

# def generate(text=None,
#              length=config.n_ctx // 2,
#              temperature=0.7,
#              top_k=40):

#     # if length > config.n_ctx:
#     #     print(f"Length cannot exceed window size. Truncated to {config.n_ctx}")
    
#     # length = min(length, config.n_ctx)
#     context_tokens = enc.encode(text) if text else None
#     start_token = enc.encoder['<|endoftext|>'] if not text else None

#     out, past = sample_sequence(
#         model=model,
#         length=length,
#         context=context_tokens,
#         start_token=start_token,
#         batch_size=1,
#         temperature=temperature,
#         top_k=top_k,
#         device=device)
    
#     out_tokens = out[:, len(context_tokens):].tolist()
#     sample = enc.decode(out_tokens[0])
    
#     return {'prompt': text, 'sample': sample, 'past': past}

# def sampler(text,
#             length=config.n_ctx // 2,
#             choose_fn=lambda x: torch.multinomial(x, num_samples=1),
#             temperature=.7):
    
#     context = torch.tensor(enc.encode(text),
#                            device=device,
#                            dtype=torch.long
#                           ).unsqueeze(0).repeat(batch_size, 1)
#     prev = context
#     past = None
    
#     with torch.no_grad():
#         for i in range(length):
#             logits, past = model(prev, past=past)
#             logits = logits[:, -1, :] / temperature
#             log_probs = torch.softmax(logits, dim=-1)
#             prev = choose_fn(log_probs)
#             yield prev, past