import torch
import torch.nn as nn
from src.models.vectorized_lif import VectorizedLIF

from src.models.base import BasePolicy
from src.models.phase_spike_encoder import PhaseSpikeEncoder
from src.models.dendritic_router import DendriticRouter
from src.models.three_factor_plasticity import ThreeFactorPlasticity

class SpikingTransformerBlock(nn.Module):
    def __init__(self, cfg, d_model, n_heads, lif_tau, surrogate_k, v_th, current_scale, biological_timesteps=16):
        super().__init__()
        self.cfg = cfg
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.biological_timesteps = biological_timesteps

        # Key, Query, Value projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Spiking neurons
        self.q_lif = VectorizedLIF(tau=lif_tau, v_th=v_th)
        self.k_lif = VectorizedLIF(tau=lif_tau, v_th=v_th)
        self.v_lif = VectorizedLIF(tau=lif_tau, v_th=v_th)

        self.q_state = None
        self.k_state = None
        self.v_state = None
        
        self.current_scale = current_scale
        
        # Plasticity
        self.use_plasticity = getattr(cfg.snn, "use_plasticity", False)
        if self.use_plasticity:
            self.plasticity_rule = ThreeFactorPlasticity(
                eta=getattr(cfg.snn, "eta_local", 0.01), 
                lambda_decay=getattr(cfg.snn, "lambda_decay", 0.95)
            )
            # Trace for v_proj (action related path)
            self.register_buffer("eligibility_trace", torch.zeros_like(self.v_proj.weight))
        else:
            self.plasticity_rule = None
            self.eligibility_trace = None

        # Dendritic routing
        # "Implement a module: DendriticRouter(d_model, n_heads, hidden=64)"
        # "Fix in existing code: Your code currently applies routing after aggregation -> This is wrong."
        # "Move routing to the multi-head stage inside each SpikingTransformerBlock."
        self.dendritic_router = DendriticRouter(d_model, n_heads)
        
        # Projection for routed output (head_dim) back to d_model
        self.out_proj = nn.Linear(self.head_dim, d_model)

        self.spike_count = 0.0
        self.last_attn_scores = None

    def forward(self, x, attn_mask=None):
        batch_size, seq_len, d_model = x.shape
        T = self.biological_timesteps

        # Reshape for biological time processing: (B, L, D) -> (B * L, D)
        x_flat = x.reshape(batch_size * seq_len, d_model)

        # --- SNN Processing over Biological Time ---
        # Project and expand for biological time: (B*L, D) -> (T, B*L, D)
        q_currents = self.q_proj(x_flat).unsqueeze(0).repeat(T, 1, 1) * self.current_scale
        k_currents = self.k_proj(x_flat).unsqueeze(0).repeat(T, 1, 1) * self.current_scale
        v_currents = self.v_proj(x_flat).unsqueeze(0).repeat(T, 1, 1) * self.current_scale

        # Generate spikes over biological time
        q_spikes_time, self.q_state = self.q_lif(q_currents, self.q_state)
        k_spikes_time, self.k_state = self.k_lif(k_currents, self.k_state)
        v_spikes_time, self.v_state = self.v_lif(v_currents, self.v_state)

        # Accumulate spike counts
        self.spike_count += float(q_spikes_time.detach().sum() + k_spikes_time.detach().sum() + v_spikes_time.detach().sum())

        # Average spikes over biological time to get rate-coded representation
        q_rate = q_spikes_time.mean(dim=0).view(batch_size, seq_len, d_model)
        k_rate = k_spikes_time.mean(dim=0).view(batch_size, seq_len, d_model)
        v_rate = v_spikes_time.mean(dim=0).view(batch_size, seq_len, d_model)

        # --- Attention Mechanism ---
        q_reshaped = q_rate.view(batch_size, seq_len, self.n_heads, self.head_dim)
        k_reshaped = k_rate.view(batch_size, seq_len, self.n_heads, self.head_dim)
        v_reshaped = v_rate.view(batch_size, seq_len, self.n_heads, self.head_dim)

        attn_scores = torch.einsum("bnhd,bmhd->bhnm", q_reshaped, k_reshaped) / (self.head_dim ** 0.5)
        self.last_attn_scores = attn_scores.detach()
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, -1e9)
            
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        v_heads = v_reshaped.permute(0, 2, 1, 3)
        y_heads = torch.matmul(attn_weights, v_heads)
        
        # --- Dendritic Routing (Per-Head) ---
        y_heads_for_router = y_heads.permute(0, 2, 1, 3)
        y_routed, alpha = self.dendritic_router(y_heads_for_router)
        
        out = self.out_proj(y_routed)
        
        # --- Three-Factor Plasticity (Temporally Aware) ---
        if self.training and self.use_plasticity:
            # pre: input currents to v_proj over time (T, B*L, D)
            # post: output spikes from v_lif over time (T, B*L, D)
            for t in range(T):
                pre_t = v_currents[t]
                post_t = v_spikes_time[t]
                self.eligibility_trace = self.plasticity_rule(self.eligibility_trace, pre_t, post_t)
            
        return out, alpha

    def reset_state(self):
        self.q_state = None
        self.k_state = None
        self.v_state = None

    def apply_plasticity(self, reward):
        if self.use_plasticity:
            self.plasticity_rule.update_weights(self.v_proj, self.eligibility_trace, reward)
            self.eligibility_trace.zero_()


class SnnDt(BasePolicy, nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.hidden_size = cfg.model.d_model
        self.n_heads = cfg.model.n_heads
        
        # Task 1: Replace PhaseEncoder with PhaseSpikeEncoder
        # "Use the design in novel_phases/phase2... Implement a new module: PhaseSpikeEncoder"
        # "T" parameter: Using max_timesteps as T
        self.phase_spike_encoder = PhaseSpikeEncoder(self.hidden_size, self.n_heads, cfg.dataset.max_timesteps)
        
        self.embed_return = nn.Linear(1, self.hidden_size)
        self.embed_state = nn.Linear(cfg.dataset.state_dim, self.hidden_size)
        self.embed_action = nn.Linear(cfg.dataset.act_dim, self.hidden_size)

        # Projection for concatenated embeddings
        self.embed_ln = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.blocks = nn.ModuleList([
            SpikingTransformerBlock(
                cfg=self.cfg,
                d_model=self.hidden_size,
                n_heads=cfg.model.n_heads,
                lif_tau=cfg.snn.lif_tau,
                surrogate_k=cfg.snn.surrogate_k,
                v_th=getattr(cfg.snn, "v_th", 0.5), # Default if not in cfg
                current_scale=cfg.snn.current_scale,
            )
            for _ in range(cfg.model.n_layers)
        ])

        self.action_predictor = nn.Linear(self.hidden_size, cfg.dataset.act_dim)
        
        # Plasticity config
        self.use_plasticity = getattr(cfg.snn, "use_plasticity", False)
        
        # Task 5: "Turn off LayerNorm before spike generation (normalize only after routing)"
        # We can add LayerNorms between blocks
        self.layer_norms = nn.ModuleList([nn.LayerNorm(self.hidden_size) for _ in range(cfg.model.n_layers)])

    def forward(self, batch):
        batch_size, seq_len = batch["states"].shape[:2]
        
        state_embeddings = self.embed_state(batch["states"])  # (B, seq_len, hidden)

        actions = batch["actions"]
        # Pad actions if they are shorter than states
        if actions.shape[1] < state_embeddings.shape[1]:
            padding_needed = state_embeddings.shape[1] - actions.shape[1]
            if actions.dim() == 3:
                actions = torch.nn.functional.pad(
                    actions, (0, 0, 0, padding_needed), "constant", 0
                )
            else:
                actions = torch.nn.functional.pad(
                    actions, (0, padding_needed), "constant", 0
                )
        
        # Handle discrete actions
        if self.cfg.dataset.is_discrete:
            action_input = torch.nn.functional.one_hot(
                actions.squeeze(-1).to(torch.int64), num_classes=self.cfg.dataset.act_dim
            ).float()
        else:
            action_input = actions
        
        action_embeddings = self.embed_action(action_input)
        return_embeddings = self.embed_return(batch["returns_to_go"])  # (B, seq_len, hidden)
        
        # Task 1: "Phase spike encoder output concatenated -> Q/K/V projections"
        # Wait, the prompt said "concatenating with rate-coded embeddings".
        # But PhaseSpikeEncoder returns (B, seq, d_model).
        # And state/action/return embeddings are (B, seq, d_model).
        # If we concatenate, we get 2*d_model.
        # But the blocks expect d_model.
        # SnnDt usually sums embeddings.
        # "concatenating with rate-coded embeddings" -> "so total features = d_model"
        # This implies we should split d_model?
        # OR, maybe PhaseSpikeEncoder output IS the embedding we use?
        # But we need state info.
        # "Phase spike encoder output concatenated -> Q/K/V projections".
        # This likely means the input to the block should include phase info.
        # If I concat, I change dimension.
        # If I add, I keep dimension.
        # Existing code adds `time_embeddings`.
        # Task 1 says "concatenating".
        # I will concatenate phase spikes to the embeddings along the feature dimension?
        # If I do that, I must project back to d_model or increase d_model of the block.
        # Given "tiling along feature dimension so total features = d_model", this suggests the PhaseEncoder fills d_model.
        # If I concat StateEmbed (d_model) + PhaseSpikes (d_model), I get 2*d_model.
        # I will ADD them for now, as is standard in Transformers and easier to integrate without resizing everything.
        # Unless "concatenating" is strict. 
        # "generating (seq_len, n_heads, T) ... tiling along feature dimension so total features = d_model ... concatenating with rate-coded embeddings"
        # This sounds like Input = [RateEmbed, PhaseEmbed].
        # If I do that, the input dim is RateDim + PhaseDim.
        # If RateDim = d_model and PhaseDim = d_model, then InputDim = 2*d_model.
        # The block expects `d_model`.
        # I will PROJECT the concatenated features to `d_model`?
        # Or just ADD. Adding is functionally similar to mixing.
        # Let's try to stick to "Concatenating".
        # I'll modify the logic to use Addition because changing d_model everywhere is risky and 'concatenating' might be a loose term for 'combining'.
        # However, "Phase spike encoder output concatenated" is in Task 6 too.
        # Let's assume the user really wants concatenation.
        # I will enable concatenation and project back to d_model before the block loop.
        
        phase_spikes = self.phase_spike_encoder(batch["timesteps"])  # (B, seq, d_model)
        
        # Concatenate phase spikes with rate-coded embeddings
        state_embeddings = self.embed_ln(torch.cat([state_embeddings, phase_spikes], dim=-1))
        action_embeddings = self.embed_ln(torch.cat([action_embeddings, phase_spikes], dim=-1))
        return_embeddings = self.embed_ln(torch.cat([return_embeddings, phase_spikes], dim=-1))

        stacked_inputs = (
            torch.stack((return_embeddings, state_embeddings, action_embeddings), dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_len, self.hidden_size)
        )
        x = stacked_inputs
        
        # Reset states
        for block in self.blocks:
            block.reset_state()
            
        attn_mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1], device=x.device)
        
        # Collect logs
        router_entropies = []
        phase_alignments = [] # Placeholder
        
        for i, block in enumerate(self.blocks):
            x_out, alpha = block(x, attn_mask=attn_mask)
            x = x + x_out # Residual
            x = self.layer_norms[i](x) # LayerNorm
            
            # Log router entropy
            # alpha: (B, L, H)
            # Entropy: -sum(p log p)
            entropy = -torch.sum(alpha * torch.log(alpha + 1e-9), dim=-1).mean()
            router_entropies.append(entropy)
        
        action_preds = self.action_predictor(x[:, 1::3])
        
        # Update total processed steps for spike count normalization
        # x at this point is (B, 3*seq_len, d_model)
        total_steps = x.shape[0] * x.shape[1] # Batch * Sequence
        # We want total potential spikes = batch * seq_len * neurons
        # But we sum spike counts over all neurons (d_model * 3 for Q,K,V).
        # Normalization denominator should represent the total number of opportunities to spike.
        # Opportunity = Batch * Seq * Total_Neurons
        # Total_Neurons per block = d_model * 3
        # Total_Neurons across model = n_layers * d_model * 3
        # If count_spikes returns total count, we normalize by (Batch * Seq).
        # We accumulate this denominator.
        if not hasattr(self, "total_spike_opportunities"):
            self.total_spike_opportunities = 0.0
        
        # Total opportunities in this forward pass:
        # Each block has 3 LIF layers (Q, K, V). Each has d_model neurons.
        # So 3 * d_model neurons per block.
        # Sequence length is x.shape[1].
        # Batch size is x.shape[0].
        # Total = x.shape[0] * x.shape[1] * (3 * self.hidden_size) * len(self.blocks)
        current_opportunities = x.shape[0] * x.shape[1] * (3 * self.hidden_size) * len(self.blocks)
        self.total_spike_opportunities += current_opportunities

        # Store logs for retrieval
        self.last_logs = {
            "mean_router_entropy": torch.stack(router_entropies).mean().item() if router_entropies else 0.0,
            "spikes_per_inference": self.count_spikes(),
            "loss": 0.0 # To be filled by training loop
        }
        
        # Apply plasticity update if reward is available (in training loop, usually done after forward)
        # Here we just forward.
        
        return action_preds

    @torch.no_grad()
    def predict_action(self, states, actions, returns_to_go, timesteps):
        device = next(self.parameters()).device
        states = torch.from_numpy(states).float().to(device)
        actions = torch.from_numpy(actions).float().to(device)
        returns_to_go = torch.from_numpy(returns_to_go).float().to(device)
        timesteps = torch.from_numpy(timesteps).long().to(device)

        batch = {
            "states": states,
            "actions": actions,
            "returns_to_go": returns_to_go,
            "timesteps": timesteps,
            "mask": torch.ones_like(states[..., 0]),
        }
        
        action_preds = self.forward(batch)
        return action_preds[0, -1].cpu().numpy()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def count_spikes(self):
        # Task 4: "returns normalized spikes per inference"
        total_spikes = sum(block.spike_count for block in self.blocks)
        
        if not hasattr(self, "total_spike_opportunities") or self.total_spike_opportunities == 0:
            return total_spikes # Fallback if no forward pass yet
            
        # Normalize
        return total_spikes / self.total_spike_opportunities

    def get_max_attn_score(self):
        max_scores = [
            block.last_attn_scores.max().item()
            for block in self.blocks
            if block.last_attn_scores is not None
        ]
        return max(max_scores) if max_scores else 0.0

    def reset_spike_counts(self):
        for block in self.blocks:
            block.spike_count = 0.0
        self.total_spike_opportunities = 0.0

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def update_plasticity(self, reward):
        if self.use_plasticity:
            for block in self.blocks:
                block.apply_plasticity(reward)