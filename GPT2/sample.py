import torch
from typing import List, Optional

def top_k_logits(logits, k):
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[-1]
    return torch.where(logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits)

def sample_token(node, top_k=40, temperature=0.7):
    p = torch.softmax(top_k_logits(node.logits, 10), dim=-1)
    return int(torch.multinomial(p, num_samples=1))