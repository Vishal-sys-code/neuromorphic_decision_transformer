import numpy as np
import torch
import torch.nn as nn
import math
import transformers
import torch.nn.functional as F

import warnings
from typing import Optional, Tuple, Union

from torch.autograd import Variable


import sys


import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


thresh = 0.5
lens = 0.5
alpha = 1
warmup_ratio = 0.1


class RepBN(nn.Module):
    def __init__(self, channels):
        super(RepBN, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)  # T D L B
        x = self.bn(x) + self.alpha * x
        x = x.permute(0, 3, 2, 1)  # T B L D
        return x


class LN(nn.Module):
    def __init__(self, dim):
        super(LN, self).__init__()
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.permute(2, 1, 0, 3).contiguous()  # L B T D
        x = self.ln(x)
        x = x.permute(2, 1, 0, 3).contiguous()  # T B L D
        return x


class BN(nn.Module):
    def __init__(self, dim):
        super(BN, self).__init__()
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)  # T D L B
        x = self.bn(x)
        x = x.permute(0, 3, 2, 1)  # T B L D
        return x


class PTNorm(nn.Module):
    def __init__(self, dim, T, step, warm=0, r0=1.0):
        super(PTNorm, self).__init__()
        self.register_buffer("warm", torch.tensor(warm, dtype=torch.long))
        self.register_buffer("iter", torch.tensor(step, dtype=torch.long))
        self.register_buffer("total_step", torch.tensor(step, dtype=torch.long))
        self.r0 = r0
        self.norm1 = LN(dim)
        self.norm2 = BN(dim)

    def forward(self, x):
        if self.training:
            if self.warm > 0:
                lamda = 1.0
            elif self.total_step <= 0:
                lamda = 0.0
            else:
                lamda = self.r0 * self.iter.float() / self.total_step.float()

            x1 = self.norm1(x)
            x2 = self.norm2(x)

            x = lamda * x1 + (1 - lamda) * x2
            if self.warm > 0:
                self.warm -= 1
            if self.iter > 0:
                self.iter -= 1
        else:
            x = self.norm2(x)
        return x


class PTNorm_Advanced(nn.Module):
    def __init__(self, dim, T, step, warm=0, r0=1.0):
        super(PTNorm_Advanced, self).__init__()
        self.register_buffer("warm", torch.tensor(warm, dtype=torch.long))
        self.register_buffer("iter", torch.tensor(step, dtype=torch.long))
        self.register_buffer("total_step", torch.tensor(step, dtype=torch.long))
        self.r0 = r0
        self.norm1 = LN([T, dim])
        self.norm2 = RepBN(dim)

    def forward(self, x):
        if self.training:
            if self.warm > 0:
                lamda = 1.0
            elif self.total_step <= 0:
                lamda = 0.0
            else:
                lamda = self.r0 * self.iter.float() / self.total_step.float()
            x1 = self.norm1(x)
            x2 = self.norm2(x)

            x = lamda * x1 + (1 - lamda) * x2
            if self.warm > 0:
                self.warm -= 1
            if self.iter > 0:
                self.iter -= 1
        else:
            x = self.norm2(x)
        return x


class Norm(nn.Module):
    def __init__(self, dim, T, step, warm=0, r0=1.0, norm_type=3):
        super().__init__()
        if norm_type == 1:
            self.norm = LN(dim)
        elif norm_type == 2:
            self.norm = BN(dim)
        elif norm_type == 3:
            self.norm = PTNorm(dim, T, step, warm, r0)
        else:
            raise ValueError("Invalid norm type.")

    def forward(self, x):
        return self.norm(x)


class ActFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = (input > thresh).float()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        temp = ((input - thresh).abs() < lens).float() / (2 * lens)
        grad_input = grad_output * temp
        return grad_input


act_fun = ActFun.apply


class LIFNode(nn.Module):
    def __init__(
        self,
        act=False,
        init_thresh=1.0,
        init_decay=0.25,
    ):
        super(LIFNode, self).__init__()
        self.thresh = init_thresh
        self.decay = init_decay
        self.actFun = (
            nn.SiLU() if act else act_fun
        )
        self.spike_count = 0

    def forward(self, x):
        T = x.size(0)
        mem = x[0]
        spike = self.actFun(mem)
        self.spike_count += spike.sum()
        outputs = [spike]

        for i in range(1, T):
            mem = mem * self.decay * (1 - spike.detach()) + x[i]
            spike = self.actFun(mem)
            self.spike_count += spike.sum()
            outputs.append(spike)

        output = torch.stack(outputs, dim=0)
        return output

    def get_spike_count(self):
        return self.spike_count

    def reset_spike_count(self):
        self.spike_count = 0

class positional_spiking_attention(nn.Module):
    def __init__(
        self,
        dim,
        T,
        num_training_steps,
        heads=8,
        seq_len=64,
        norm_type=3,
        window_size=8,
    ):
        super().__init__()
        assert dim % heads == 0, f"dim {dim} should be divided by num_heads {heads}."
        self.norm_type = norm_type

        self.L = seq_len
        self.dim = dim
        self.heads = heads

        self.q_m = nn.Linear(dim, dim)
        self.q_ln = Norm(
            dim, T, step=int(num_training_steps * warmup_ratio), norm_type=self.norm_type
        )
        self.q_lif = LIFNode(act=False)

        self.k_m = nn.Linear(dim, dim)
        self.k_ln = Norm(
            dim, T, step=int(num_training_steps * warmup_ratio), norm_type=self.norm_type
        )
        self.k_lif = LIFNode(act=False)

        self.v_m = nn.Linear(dim, dim)
        self.v_ln = Norm(
            dim, T, step=int(num_training_steps * warmup_ratio), norm_type=self.norm_type
        )
        self.v_lif = LIFNode(act=False)

        self.attn_lif = LIFNode(act=False)

        self.last_m = nn.Linear(dim, dim)
        self.last_ln = Norm(
            dim, T, step=int(num_training_steps * warmup_ratio), norm_type=self.norm_type
        )
        if self.norm_type == 3:
            self.last_ln.norm.norm1.is_shortcut = True
            self.last_ln.norm.norm2.is_shortcut = True

        self.first_lif = LIFNode(act=False)

        local_window_size = window_size
        self.pos_bias = nn.Parameter(torch.ones(seq_len, seq_len), requires_grad=True)
        self.register_buffer(
            "local_mask", self.create_local_mask(seq_len, local_window_size)
        )

    @staticmethod
    def create_local_mask(seq_len, local_window_size):
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool)
        mask = torch.tril(mask, local_window_size - 1)
        mask = torch.triu(mask, -local_window_size + 1)
        return mask


    def forward(self, x, attention_mask=None):
        pos_bias = self.pos_bias
        L = x.size(2)
        pos_bias = pos_bias[:L, :L] * self.local_mask[:L, :L]
        pos_bias = pos_bias.unsqueeze(0).unsqueeze(0)
        mask = torch.tril(torch.ones(L, L, device=x.device)).view(1, 1, L, L)
        pos_bias = pos_bias.masked_fill(mask == 0, 0)
        if attention_mask is not None:
            batch_size = x.shape[1]
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            pos_bias = pos_bias.masked_fill(attention_mask == 0, 0)
        x = self.first_lif(x)
        T, B, L, D = x.shape
        q_m_out = self.q_m(x)
        q_m_out = self.q_ln(q_m_out)
        q_m_out = self.q_lif(q_m_out)
        q = (
            q_m_out.reshape(T, B, L, self.heads, D // self.heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )
        q = q.permute(1, 2, 3, 0, 4).reshape(
            B, self.heads, L, T * D // self.heads
        )

        k_m_out = self.k_m(x)
        k_m_out = self.k_ln(k_m_out)
        k_m_out = self.k_lif(k_m_out)

        v_m_out = self.v_m(x)
        v_m_out = self.v_ln(v_m_out)
        v_m_out = self.v_lif(v_m_out)

        kv = (
            (k_m_out * v_m_out)
            .reshape(T, B, L, self.heads, D // self.heads)
            .permute(1, 3, 2, 0, 4)
        )
        kv = kv.reshape(B, self.heads, L, T * D // self.heads)
        attn = torch.einsum("bhij,bhjd->bhid", pos_bias, kv)
        x = q * attn
        x = (
            x.reshape(B, self.heads, L, T, D // self.heads)
            .permute(3, 0, 2, 1, 4)
            .reshape(T, B, L, D)
        )
        x = self.attn_lif(x)

        x = self.last_m(x)
        x = self.last_ln(x)
        return x


class mlp(nn.Module):
    def __init__(
        self,
        T,
        num_training_steps,
        in_features,
        hidden_features=None,
        out_features=None,
        norm_type=3,
    ):
        super().__init__()
        self.norm_type = norm_type
        out_features = out_features or in_features
        hidden_features = hidden_features
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.ln1 = Norm(
            hidden_features,
            T,
            step=int(num_training_steps * warmup_ratio),
            norm_type=self.norm_type,
        )
        self.lif1 = LIFNode(act=False)

        self.fc2 = nn.Linear(hidden_features, out_features)
        self.ln2 = Norm(
            out_features,
            T,
            step=int(num_training_steps * warmup_ratio),
            norm_type=self.norm_type,
        )
        if self.norm_type == 3:
            self.ln2.norm.norm1.is_shortcut = True
            self.ln2.norm.norm2.is_shortcut = True
        self.lif2 = LIFNode(act=False)

    def forward(self, x):
        x = self.lif1(x)
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.lif2(x)
        x = self.fc2(x)
        x = self.ln2(x)
        return x


class block(nn.Module):
    def __init__(
        self,
        drop_dpr,
        dim,
        T,
        num_training_steps,
        heads=8,
        qkv_bias=False,
        seq_len=64,
        attn_type=3,
        norm_type=3,
        window_size=8,
    ):
        super().__init__()
        self.attn = positional_spiking_attention(
            dim=dim,
            T=T,
            heads=heads,
            seq_len=seq_len,
            num_training_steps=num_training_steps,
            norm_type=norm_type,
            window_size=window_size,
        )
        self.mlp = mlp(
            T=T,
            in_features=dim,
            hidden_features=dim * 4,
            out_features=dim,
            num_training_steps=num_training_steps,
            norm_type=norm_type,
        )

    def forward(self, x, attention_mask=None):
        x = x + self.attn(x, attention_mask=attention_mask)
        x = x + self.mlp(x)
        return x

class new_spikformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.T = config.T
        dim = config.n_embd
        heads = config.n_head
        seq_len = config.ctx_len
        attn_type = config.attn_type
        self.norm_type = config.norm_type
        num_training_steps = config.num_training_steps
        window_size = config.window_size
        self.drop_path_rate = 0.05
        self.blocks = nn.ModuleList(
            [
                block(
                    drop_dpr=self.drop_path_rate * float(idx) / config.n_layer,
                    dim=dim,
                    T=self.T,
                    heads=heads,
                    seq_len=seq_len,
                    attn_type=attn_type,
                    num_training_steps=num_training_steps,
                    norm_type=self.norm_type,
                    window_size=window_size,
                )
                for idx in range(config.n_layer)
            ]
        )
        self.last_ln = Norm(
            dim, self.T, step=int(num_training_steps * warmup_ratio), norm_type=self.norm_type
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            if hasattr(m, "is_shortcut") and m.is_shortcut:
                nn.init.constant_(m.weight, thresh / (2**0.5))
            else:
                nn.init.constant_(m.weight, thresh)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x, attention_mask=None):
        x = x.repeat(
            tuple([self.T] + torch.ones(len(x.size()), dtype=int).tolist())
        )
        for i, decoder in enumerate(self.blocks):
            x = decoder(x, attention_mask=attention_mask)
        x = self.last_ln(x)
        x = x.mean(0)
        return x


from types import SimpleNamespace


class SpikeDecisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        T = 4
        attn_type = 3
        norm_type = 1
        window_size = 8
        global warmup_ratio
        warmup_ratio = config.warmup_ratio
        self.T = T
        num_training_steps = config.num_training_steps
        self.dropout = 0.1
        config = dict(
            vocab_size=1,
            n_embd=config.n_embd,
            n_ctx=config.n_positions,
            n_layer=config.n_layer,
            n_head=config.n_head,
            ctx_len=config.n_positions,
            T=self.T,
            attn_type=attn_type,
            num_training_steps=num_training_steps,
            norm_type=norm_type,
            window_size=window_size,
        )
        config = SimpleNamespace(**config)
        self.transformer = new_spikformer(config)

    def forward(
        self,
        srcs: torch.Tensor,
        attention_mask=None,
    ):
        outputs = self.transformer(srcs, attention_mask=attention_mask)
        return outputs


class TrajectoryModel(nn.Module):

    def __init__(self, state_dim, act_dim, max_length=None):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length

    def forward(self, states, actions, rewards, masks=None, attention_mask=None):
        return None, None, None

    def get_action(self, states, actions, rewards, **kwargs):
        return torch.zeros_like(actions[-1])

class DecisionSpikeFormer(TrajectoryModel):

    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,
            n_embd=hidden_size,
            n_positions=max_length,
            **kwargs
        )

        self.transformer = SpikeDecisionTransformer(config)
        self.embed = nn.Linear(self.state_dim + self.act_dim + 1, hidden_size)
        self.embed_ln = nn.LayerNorm(hidden_size)
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )

    def forward(self, states, actions, returns_to_go, timesteps, attention_mask=None, output_attentions=False):

        batch_size, seq_length = states.shape[0], states.shape[1]

        actions = torch.cat([torch.zeros_like(actions[:,0:1]), actions[:,:-1]], dim=1)
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=states.device)

        embeddings = self.embed(torch.cat([states, actions, returns_to_go], dim=-1))
        stacked_inputs = self.embed_ln(embeddings)

        transformer_outputs = self.transformer(
            stacked_inputs,
            attention_mask=attention_mask,
        )
        x = transformer_outputs
        action_preds = self.predict_action(x)

        return None, action_preds, None

    def get_action(self, states, actions, returns_to_go, timesteps,**kwargs):
        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim),
                             device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1
            ).to(dtype=torch.long)
        else:
            attention_mask = None

        _, action_preds, _ = self.forward(
            states, actions, returns_to_go, timesteps, attention_mask=attention_mask, **kwargs)

        return action_preds[0,-1]

    def get_total_spike_count(self):
        total_spikes = 0
        for module in self.modules():
            if isinstance(module, LIFNode):
                total_spikes += module.get_spike_count()
        return total_spikes

    def reset_total_spike_count(self):
        for module in self.modules():
            if isinstance(module, LIFNode):
                module.reset_spike_count()