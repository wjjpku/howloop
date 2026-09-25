import os,json,time,hashlib,argparse,random,itertools
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from reasoning_loop.graph_path_depth_circuit import load_checkpoint,fixed_depth_batch
R=Path(__file__).resolve().parent.parent
class AffineJ(torch.nn.Module):
 def __init__(self,width):
  super().__init__();self.delta=torch.nn.Linear(width,width);torch.nn.init.zeros_(self.delta.weight);torch.nn.init.zeros_(self.delta.bias)
 def forward(self,x):
  h=x[:,-1:];return torch.cat((x[:,:-1],h+self.delta(h)),1)
def readout(m,x):return m.unembed(m.ln_final(x[:,-1]))[:,:10]
def save_json(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
