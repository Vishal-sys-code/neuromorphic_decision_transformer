import copy
import math

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
            self.log_std_min = -20
            self.log_std_max = 2

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        if self.is_discrete:
            return self.logits(x)
        else:
            mu = self.mu(x)
            log_std = torch.clamp(self.log_std_linear(x), self.log_std_min, self.log_std_max)
            return mu, log_std

    def evaluate(self, state, epsilon=1e-6):
        if self.is_discrete:
            logits = self.forward(state)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action).unsqueeze(-1)
            return action, log_prob
        else:
            mu, log_std = self.forward(state)
            std = log_std.exp()
            dist = Normal(mu, std)
            e = dist.rsample()
            action = torch.tanh(e)
            log_prob = (dist.log_prob(e) - torch.log(1 - action.pow(2) + epsilon)).sum(1, keepdim=True)
            return action, log_prob

    def get_action(self, state):
        if self.is_discrete:
            logits = self.forward(state)
            dist = Categorical(logits=logits)
            return dist.sample()
        else:
            mu, log_std = self.forward(state)
            std = log_std.exp()
            dist = Normal(mu, std)
            e = dist.rsample()
            return torch.tanh(e)

    def get_det_action(self, state):
        if self.is_discrete:
            logits = self.forward(state)
            return torch.argmax(logits, dim=-1)
        else:
            mu, _ = self.forward(state)
            return torch.tanh(mu)


class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256, is_discrete=False):
        super(Critic, self).__init__()
        self.is_discrete = is_discrete
        if self.is_discrete:
            # For discrete actions, we use an embedding layer.
            # The input to fc1 will be state_size + embedding_dim.
            # Let's use hidden_size as the embedding dimension.
            self.action_embedding = nn.Embedding(action_size, hidden_size)
            self.fc1 = nn.Linear(state_size + hidden_size, hidden_size)
        else:
            self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        if self.is_discrete:
            # action is an index
            action_emb = self.action_embedding(action.long().squeeze(-1))
            x = torch.cat((state, action_emb), dim=-1)
        else:
            x = torch.cat((state, action), dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class CQL(BasePolicy, nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = cfg.training.device
        self.gamma = 0.99
        self.tau = cfg.cql.tau
        self.target_entropy = -cfg.dataset.act_dim

        self.log_alpha = torch.tensor([0.0], requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp().detach()

        self.is_discrete = 'CartPole' in cfg.env or 'Acrobot' in cfg.env or 'MountainCar' in cfg.env

        self.actor = Actor(cfg.dataset.state_dim, cfg.dataset.act_dim, cfg.cql.hidden_size, is_discrete=self.is_discrete).to(self.device)
        self.critic1 = Critic(cfg.dataset.state_dim, cfg.dataset.act_dim, cfg.cql.hidden_size, is_discrete=self.is_discrete).to(self.device)
        self.critic2 = Critic(cfg.dataset.state_dim, cfg.dataset.act_dim, cfg.cql.hidden_size, is_discrete=self.is_discrete).to(self.device)
        self.critic1_target = Critic(cfg.dataset.state_dim, cfg.dataset.act_dim, cfg.cql.hidden_size, is_discrete=self.is_discrete).to(self.device)
        self.critic2_target = Critic(cfg.dataset.state_dim, cfg.dataset.act_dim, cfg.cql.hidden_size, is_discrete=self.is_discrete).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        lr = float(cfg.training.lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

        # CQL specific
        self.with_lagrange = cfg.cql.with_lagrange
        self.temp = cfg.cql.temperature
        self.cql_weight = cfg.cql.cql_weight
        self.target_action_gap = cfg.cql.target_action_gap
        self.cql_log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.cql_alpha_optimizer = optim.Adam([self.cql_log_alpha], lr=lr)

    def forward(self, batch):
        pass

    def learn(self, batch):
        states, actions, rewards, next_states, dones = (
            batch["states"],
            batch["actions"],
            batch["rewards"],
            batch["next_states"],
            batch["dones"],
        )

        # Update actor and alpha
        actor_loss, log_pis = self.calc_policy_loss(states, self.alpha)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha.exp() * (log_pis + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()

        # Update critics
        with torch.no_grad():
            next_action, new_log_pi = self.actor.evaluate(next_states)
            q1_target_next = self.critic1_target(next_states, next_action)
            q2_target_next = self.critic2_target(next_states, next_action)
            q_target_next = torch.min(q1_target_next, q2_target_next) - self.alpha * new_log_pi
            q_targets = rewards + self.gamma * (1 - dones) * q_target_next

        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        critic1_loss = F.mse_loss(q1, q_targets)
        critic2_loss = F.mse_loss(q2, q_targets)

        # CQL loss
        cql1_loss, cql2_loss = self.calc_cql_loss(states, actions, q1, q2)
        
        # Lagrange multiplier for CQL
        cql_alpha_loss, cql_alpha = torch.tensor(0.0), torch.tensor(0.0)
        if self.with_lagrange:
            cql_alpha = torch.clamp(self.cql_log_alpha.exp(), min=0.0, max=1000000.0)
            cql1_loss = cql_alpha * (cql1_loss - self.target_action_gap)
            cql2_loss = cql_alpha * (cql2_loss - self.target_action_gap)
            cql_alpha_loss = (-cql1_loss - cql2_loss) * 0.5
            self.cql_alpha_optimizer.zero_grad()
            cql_alpha_loss.backward(retain_graph=True)
            self.cql_alpha_optimizer.step()

        total_critic1_loss = critic1_loss + cql1_loss
        total_critic2_loss = critic2_loss + cql2_loss

        self.critic1_optimizer.zero_grad()
        total_critic1_loss.backward(retain_graph=True)
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        total_critic2_loss.backward()
        self.critic2_optimizer.step()

        # Soft update target networks
        self.soft_update(self.critic1, self.critic1_target)
        self.soft_update(self.critic2, self.critic2_target)

        value_loss = (total_critic1_loss + total_critic2_loss) / 2.0

        return {
            "value_loss": value_loss.item(),
            "policy_loss": actor_loss.item(),
            "critic1_loss": critic1_loss.item(),
            "critic2_loss": critic2_loss.item(),
            "cql1_loss": cql1_loss.item(),
            "cql2_loss": cql2_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "cql_alpha_loss": cql_alpha_loss.item(),
            "alpha": self.alpha.item(),
            "cql_alpha": cql_alpha.item(),
        }

    def calc_policy_loss(self, states, alpha):
        actions_pred, log_pis = self.actor.evaluate(states)
        q1 = self.critic1(states, actions_pred)
        q2 = self.critic2(states, actions_pred)
        min_q = torch.min(q1, q2)
        actor_loss = ((alpha * log_pis - min_q)).mean()
        return actor_loss, log_pis

    def calc_cql_loss(self, states, actions, q1, q2):
        if self.is_discrete:
            action_size = self.actor.logits.out_features
            
            all_actions = torch.arange(action_size, device=self.device).unsqueeze(0).repeat(states.shape[0], 1)
            states_repeated = states.unsqueeze(1).repeat(1, action_size, 1)
            
            all_actions_flat = all_actions.view(-1, 1)
            states_flat = states_repeated.view(-1, states.shape[1])

            q1_all = self.critic1(states_flat, all_actions_flat).view(states.shape[0], -1)
            q2_all = self.critic2(states_flat, all_actions_flat).view(states.shape[0], -1)

            cql1_loss = (torch.logsumexp(q1_all / self.temp, dim=1).mean() * self.cql_weight * self.temp) - q1.mean()
            cql2_loss = (torch.logsumexp(q2_all / self.temp, dim=1).mean() * self.cql_weight * self.temp) - q2.mean()
        else:
            random_actions = torch.FloatTensor(q1.shape[0] * 10, actions.shape[-1]).uniform_(-1, 1).to(self.device)
            num_repeat = int(random_actions.shape[0] / states.shape[0])
            temp_states = states.unsqueeze(1).repeat(1, num_repeat, 1).view(states.shape[0] * num_repeat, states.shape[1])
            
            # Get values for random actions
            random_values1 = self.critic1(temp_states, random_actions).reshape(states.shape[0], num_repeat, 1)
            random_values2 = self.critic2(temp_states, random_actions).reshape(states.shape[0], num_repeat, 1)

            # Get values for policy actions
            with torch.no_grad():
                policy_actions, log_pis = self.actor.evaluate(temp_states)
            policy_values1 = self.critic1(temp_states, policy_actions).reshape(states.shape[0], num_repeat, 1)
            policy_values2 = self.critic2(temp_states, policy_actions).reshape(states.shape[0], num_repeat, 1)

            cat_q1 = torch.cat([random_values1, policy_values1 - log_pis.detach().reshape(states.shape[0], num_repeat, 1)], 1)
            cat_q2 = torch.cat([random_values2, policy_values2 - log_pis.detach().reshape(states.shape[0], num_repeat, 1)], 1)

            cql1_loss = (torch.logsumexp(cat_q1 / self.temp, dim=1).mean() * self.cql_weight * self.temp) - q1.mean()
            cql2_loss = (torch.logsumexp(cat_q2 / self.temp, dim=1).mean() * self.cql_weight * self.temp) - q2.mean()
        return cql1_loss, cql2_loss

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