'''
    adapted from code by TaeHwan Jung(@graykode)
    Original Paper and repository here : https://github.com/openai/gpt-2
    GPT2 Pytorch Model : https://github.com/huggingface/pytorch-pretrained-BERT
'''
import torch
import torch.nn.functional as F
from typing import List, Optional
from tqdm import trange
from functools import partial, lru_cache

def cachedproperty(func=None, *, maxsize=128):
    if func == None:
        return partial(cachedproperty, maxsize=maxsize)
    return property(lru_cache(maxsize=maxsize)(func))

def top_k_logits(logits, k):
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[-1]
    return torch.where(logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits)

class SampleNode:
    '''Represents a single token in the GPT-2 model
    with a history, and potentially multiple futures.'''
    def __init__(self, 
                model: 'GPT2LMHeadModel',
                token_ids: List[int]=None,
                parent: Optional['SampleNode']=None,
                *args, **kwargs):
        
        self.model = model
        *rest_ids, last_id = token_ids
        self.token_id = torch.tensor([[last_id]], dtype=torch.long)
        
        if len(rest_ids) > 0:
            cls = type(self)
            self.parent = cls(model, rest_ids, parent)
        else:
            self.parent = parent
        
        self._hidden_state = None
            
        if self.parent is not None:
            self.parent.children.append(self)      
            
        self.children = []
    
    @cachedproperty(maxsize=1)
    def past(self) -> List[torch.Tensor]:
        if self.parent is None:
            return [torch.zeros(2, 1, 12, 0, 64) for _ in range(12)]
        return [torch.cat([x[...,-1023:,:], y], axis=-2) 
                for x, y in zip(self.parent.past, self.parent.hidden_state)]
    
    @property
    def hidden_state(self) -> List[torch.Tensor]:
        if self._hidden_state is None:
            self.next_token_logits
        return self._hidden_state
    
    @hidden_state.setter
    def hidden_state(self, value):
        if self._hidden_state is None:
            self._hidden_state = value
            return
    
    @cachedproperty(maxsize=1)
    @torch.no_grad()
    def next_token_logits(self) -> torch.Tensor:
        past = None if self.parent is None else self.past
        logits, present = self.model(self.token_id, past=past)
        self.hidden_state = [p[...,-1:,:] for p in present]
        return logits[0, -1, :]
    
    def random_new_child(self, top_k=40, temperature=.7):
        log_prob = torch.softmax(
            top_k_logits(self.next_token_logits / temperature, top_k), dim=-1)
        token_id = torch.multinomial(log_prob, num_samples=1)
        
        return self.new_descendents([int(token_id[0])])
    
    def random_new_descendents(self, n, top_k=40, temperature=.7):
        if n < 1:
            raise ValueError('n must be greater than or equal to 1')
        child = self.random_new_child(top_k=top_k, temperature=temperature)
        if n == 1:
            return child
        else:
            return child.random_new_descendents(n-1, top_k=top_k, temperature=temperature)
        
    def new_descendents(self, token_ids) -> 'SampleNode':
        cls = type(self)
        first, *rest = token_ids
        for child in self.children:
            if int(child.token_id[0][0]) == first:
                if len(rest) > 0:
                    child = child.new_descendents(rest)
                return child
        child = cls(self.model, token_ids, self)
        return child
    
    def tokens(self, n=10):
        if self.parent is None or n == 0:
            p_tokens = []
        else:
            p_tokens = self.parent.tokens(n-1)
            
        return p_tokens + [int(self.token_id[0])]
    
    @property
    def root(self):
        if self.parent is None: return self
        return self.parent.root
    
    @cachedproperty
    def cumulative_logit(self):
        cl = self.logit
        if self.parent is not None:
            cl += self.parent.logit
        
        return cl
        
    @cachedproperty
    def logit(self):
        if self.parent == None:
            return 0
        else:
            return self.parent.next_token_logits[self.token_id[0][0]]