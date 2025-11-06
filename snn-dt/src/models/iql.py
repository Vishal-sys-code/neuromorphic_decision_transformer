import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical

from src.models.base import BasePolicy


class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256, is_discrete=False):
        super(Actor, self).__init__()
        self.is_discrete = is_discrete
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        if is_discrete:
            self.logits = nn.Linear(hidden_size, action_size)
        else:
            self.mu = nn.Linear(hidden_size, action_size)
            self.log_std_linear = nn.Linear(hidden_size, action_size)
            self.log_std_min = -10
            self.log_std_max = 2

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        if self.is_discrete:
            return self.logits(x)
        else:
            mu = torch.tanh(self.mu(x))
            log_std = torch.clamp(self.log_std_linear(x), self.log_std_min, self.log_std_max)
            return mu, log_std

    def evaluate(self, state):
        if self.is_discrete:
            logits = self.forward(state)
            dist = Categorical(logits=logits)
            return dist.sample(), dist
        else:
            mu, log_std = self.forward(state)
            std = log_std.exp()
            dist = Normal(mu, std)
            return dist.rsample(), dist

    def get_action(self, state):
        if self.is_discrete:
            logits = self.forward(state)
            dist = Categorical(logits=logits)
            return dist.sample()
        else:
            mu, log_std = self.forward(state)
            std = log_std.exp()
            dist = Normal(mu, std)
            return dist.rsample()

    def get_det_action(self, state):
        if self.is_discrete:
            logits = self.forward(state)
            return torch.argmax(logits, dim=-1)
        else:
            mu, _ = self.forward(state)
            return mu


class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Value(nn.Module):
    def __init__(self, state_size, hidden_size=256):
        super(Value, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def loss_fn(diff, expectile=0.8):
    weight = torch.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff ** 2)


class IQL(BasePolicy, nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = cfg.training.device
        self.gamma = 0.99
        self.tau = cfg.iql.tau
        self.temperature = cfg.iql.temperature
        self.expectile = cfg.iql.expectile

        self.actor = Actor(cfg.dataset.state_dim, cfg.dataset.act_dim, cfg.iql.hidden_size).to(self.device)
        self.critic1 = Critic(cfg.dataset.state_dim, cfg.dataset.act_dim, cfg.iql.hidden_size).to(self.device)
        self.critic2 = Critic(cfg.dataset.state_dim, cfg.dataset.act_dim, cfg.iql.hidden_size).to(self.device)
        self.value_net = Value(cfg.dataset.state_dim, cfg.iql.hidden_size).to(self.device)

        self.critic1_target = Critic(cfg.dataset.state_dim, cfg.dataset.act_dim, cfg.iql.hidden_size).to(self.device)
        self.critic2_target = Critic(cfg.dataset.state_dim, cfg.dataset.act_dim, cfg.iql.hidden_size).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.training.lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=cfg.training.lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=cfg.training.lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=cfg.training.lr)

    def forward(self, batch):
        # IQL has a custom learn method, so forward is a no-op for now
        pass

    def learn(self, batch):
        states, actions, rewards, next_states, dones = (
            batch["states"],
            batch["actions"],
            batch["rewards"],
            batch["next_states"],
            batch["dones"],
        )

        # Value loss
        with torch.no_grad():
            q1 = self.critic1_target(states, actions)
            q2 = self.critic2_target(states, actions)
            min_q = torch.min(q1, q2)
        value = self.value_net(states)
        value_loss = loss_fn(min_q - value, self.expectile).mean()
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Policy loss
        with torch.no_grad():
            v = self.value_net(states)
            exp_a = torch.exp((min_q - v) * self.temperature)
            exp_a = torch.min(exp_a, torch.tensor(100.0, device=self.device))
        _, dist = self.actor.evaluate(states)
        log_probs = dist.log_prob(actions)
        actor_loss = -(exp_a * log_probs).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Q-function loss
        with torch.no_grad():
            next_v = self.value_net(next_states)
            q_target = rewards + self.gamma * (1 - dones) * next_v
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        critic1_loss = F.mse_loss(q1, q_target)
        critic2_loss = F.mse_loss(q2, q_target)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Soft update target networks
        self.soft_update(self.critic1, self.critic1_target)
        self.soft_update(self.critic2, self.critic2_target)

        return {
            "policy_loss": actor_loss.item(),
            "critic1_loss": critic1_loss.item(),
            "critic2_loss": critic2_loss.item(),
            "value_loss": value_loss.item(),
        }

    def soft_update(self, local, target):
        for target_param, local_param in zip(target.parameters(), local.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def predict_action(self, state, deterministic=True):
        state = torch.from_numpy(state).float().to(self.device)
        with torch.no_grad():
            if deterministic:
                action = self.actor.get_det_action(state)
            else:
                action = self.actor.get_action(state)
        return action.cpu().numpy()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def count_spikes(self):
        return 0

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)