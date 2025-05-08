import torch
from src.models.snn_dt import SNNDecisionTransformer

def test_snn_dt_forward_backward():
    torch.manual_seed(0)
    # small toy config
    config = dict(
        state_dim=4,
        act_dim=2,
        hidden_size=32,
        max_length=10,
        n_layer=2,
        n_head=1,
        n_inner=64,
        time_window=5
    )
    config['n_ctx'] = config['max_length']
    model = SNNDecisionTransformer(**config)
    batch = 3
    # dummy trajectory batch
    states = torch.randn(batch, config['max_length'], config['state_dim'], requires_grad=False)
    actions = torch.randint(0, config['act_dim'], (batch, config['max_length'], config['act_dim'])).float()
    returns_to_go = torch.randn(batch, config['max_length'], 1)
    timesteps = torch.arange(config['max_length']).unsqueeze(0).repeat(batch,1)

    # dummy rewards tensor
    rewards = torch.zeros(batch, config['max_length'], 1)
    # forward
    _state_preds, action_preds, _return_preds = model(states, actions, rewards, returns_to_go, timesteps)
    assert action_preds.shape == (batch, config['max_length'], config['act_dim'])

    # backward
    loss = action_preds.mean()
    loss.backward()
    # confirm at least one grad is non-zero
    total_grad = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
    assert total_grad > 0