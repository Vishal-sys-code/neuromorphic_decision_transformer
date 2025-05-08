import torch
from src.models.snn_dt_patch import SNNDecisionTransformer

def test_snn_dt_gpt2_patch():
    torch.manual_seed(0)
    cfg = dict(
        state_dim=4, act_dim=2,
        hidden_size=32, max_length=10,
        n_ctx=10,
        max_ep_len=1000,
        action_tanh=False,
        n_layer=2, n_head=1, n_inner=64,
        time_window=5
    )
    model = SNNDecisionTransformer(**cfg)
    B, S = 2, cfg['max_length']
    states = torch.randn(B, S, cfg['state_dim'])
    actions = torch.randint(0, cfg['act_dim'], (B, S, cfg['act_dim'])).float()
    returns_to_go = torch.randn(B, S, 1)
    timesteps = torch.arange(S).unsqueeze(0).repeat(B,1)
    # forward
    state_preds, action_preds, return_preds = model(
        states, actions, None, returns_to_go, timesteps
    )
    assert action_preds.shape == (B, S, cfg['act_dim'])
    # backward
    loss = action_preds.mean()
    loss.backward()
    # check at least one grad in spiking attention
    grads = [p.grad for p in model.transformer.h[0].attn.snn_attn.parameters() if p.grad is not None]
    assert sum(g.abs().sum().item() for g in grads) > 0