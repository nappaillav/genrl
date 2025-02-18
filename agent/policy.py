import torch.nn as nn
import torch

import tools.utils as utils
import agent.dreamer_utils as common
from collections import OrderedDict
import numpy as np

from tools.genrl_utils import *

def stop_gradient(x):
  return x.detach()

Module = nn.Module 

class BCPolicy(Module):
  def __init__(self, config, act_spec, feat_size, name='', embed_dim=1536, goal_condition=None):
    super().__init__()
    self.name = name
    self.cfg = config
    self.act_spec = act_spec
    self._use_amp = (config.precision == 16)
    self.device = config.device
    self.embed_dim = embed_dim
    self.gc = goal_condition
    inp_size = feat_size + embed_dim if goal_condition else 0
    
    if getattr(self.cfg, 'discrete_actions', False):
      self.cfg.actor.dist = 'onehot'

    self.actor_grad = getattr(self.cfg, f'{self.name}_actor_grad'.strip('_'))
    
    inp_size = feat_size + embed_dim
  
    self.actor = common.MLP(inp_size, act_spec.shape[0], **self.cfg.actor)
    self.actor_opt = common.Optimizer('actor', self.actor.parameters(), **self.cfg.actor_opt, use_amp=self._use_amp)

  def update(self, world_model, start, batch, task_cond=None):
    metrics = {}
    with common.RequiresGrad(self.actor):
        with torch.cuda.amp.autocast(enabled=self._use_amp):
            action, entropy = self.OnestepBC(world_model, start, task_cond)
            mse_loss = ((batch['action'].reshape(-1, action.shape[-1]) - action)**2).mean() 
            actor_loss = mse_loss + self.cfg.actor_ent * entropy
            metrics.update(self.actor_opt(actor_loss, self.actor.parameters()))
            metrics.update({"Policy_MSE" : mse_loss.item(), "Policy_Entropy" : entropy.item()})   
    return { f'{self.name}_{k}'.strip('_') : v for k,v in metrics.items() }
    
  def OnestepBC(self, world_model, start, task_cond=None, eval_policy=False):
    flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
    start = {k: flatten(v) for k, v in start.items()}
    start['feat'] = world_model.rssm.get_feat(start)
    inp = start['feat'] if task_cond is None else torch.cat([start['feat'], task_cond], dim=-1)
    policy_dist = self.actor(stop_gradient(inp))
    return policy_dist.sample(), policy_dist.entropy().mean() 