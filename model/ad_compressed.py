import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import pack, rearrange, repeat

from env import map_dark_states, map_dark_states_inverse


class CompressionEncoder(nn.Module):
    """
    Transformer encoder that compresses context history into fixed number of latent tokens.
    
    The encoder takes context embeddings (from previous latents and/or new context) plus
    learnable query tokens, and outputs n_latent compressed tokens representing the history.
    """
    def __init__(self, config):
        super(CompressionEncoder, self).__init__()
        
        self.config = config
        self.device = config['device']
        self.n_latent = config['n_latent']
        
        tf_n_embd = config['tf_n_embd']
        encoder_n_head = config.get('encoder_n_head', config.get('tf_n_head', 4))
        encoder_n_layer = config.get('encoder_n_layer', config.get('tf_n_layer', 4))
        encoder_dim_feedforward = config.get('encoder_dim_feedforward', tf_n_embd * 4)
        
        # Learnable query tokens that act as "compression slots"
        # These are what the encoder outputs - they learn to extract relevant info
        self.query_tokens = nn.Parameter(torch.randn(1, self.n_latent, tf_n_embd))
        nn.init.trunc_normal_(self.query_tokens, std=0.02)
        
        # Positional embedding for encoder input (context + queries)
        max_encoder_len = config.get('max_encoder_length', 512)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_encoder_len, tf_n_embd))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=tf_n_embd,
            nhead=encoder_n_head,
            dim_feedforward=encoder_dim_feedforward,
            activation='gelu',
            batch_first=True,
            dropout=config.get('tf_dropout', 0.1),
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_n_layer)
        
    def forward(self, context_embeddings):
        """
        Args:
            context_embeddings: [batch, seq_len, tf_n_embd]
                Can be mixture of: previous latent tokens + new context embeddings
        
        Returns:
            latent_tokens: [batch, n_latent, tf_n_embd]
                Compressed representation of the context
        """
        batch_size = context_embeddings.size(0)
        
        # Expand query tokens for batch
        query_tokens = repeat(self.query_tokens, '1 n d -> b n d', b=batch_size)
        
        # Concatenate context with query tokens
        encoder_input = torch.cat([context_embeddings, query_tokens], dim=1)  # [B, L+n_latent, D]
        
        # Add positional embeddings
        seq_len = encoder_input.size(1)
        encoder_input = encoder_input + self.pos_embedding[:, :seq_len, :]
        
        # Apply transformer encoder
        encoder_output = self.transformer_encoder(encoder_input)  # [B, L+n_latent, D]
        
        # Extract only the query token positions (last n_latent tokens)
        # These contain the compressed information
        latent_tokens = encoder_output[:, -self.n_latent:, :]  # [B, n_latent, D]
        
        return latent_tokens


class CompressedAD(nn.Module):
    """
    Algorithm Distillation model with hierarchical compression mechanism.
    
    Uses a CompressionEncoder to compress old context into latent tokens when
    sequence length exceeds maximum, then a decoder processes latent tokens +
    recent uncompressed context to predict actions.
    """
    def __init__(self, config):
        super(CompressedAD, self).__init__()
        
        self.config = config
        self.device = config['device']
        self.n_transit = config['n_transit']
        self.max_seq_length = config['decoder_max_seq_length']  # Max length for decoder
        self.n_latent = config['n_latent']
        self.mixed_precision = config['mixed_precision']
        self.grid_size = config['grid_size']
        
        tf_n_embd = config['tf_n_embd']
        tf_n_head = config.get('tf_n_head', 4)
        tf_n_layer = config.get('tf_n_layer', 4)
        tf_dim_feedforward = config.get('tf_dim_feedforward', tf_n_embd * 4)
        
        # Compression encoder
        self.encoder = CompressionEncoder(config)
        
        # Embeddings (shared between encoder and decoder)
        self.embed_context = nn.Linear(config['dim_states'] * 2 + config['num_actions'] + 1, tf_n_embd)
        self.embed_query_state = nn.Embedding(config['grid_size'] * config['grid_size'], tf_n_embd)
        
        # Decoder positional embedding
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.max_seq_length, tf_n_embd))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        
        # Register causal mask buffer
        self.register_buffer('causal_mask', None, persistent=False)
        
        # Decoder transformer
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=tf_n_embd,
            nhead=tf_n_head,
            dim_feedforward=tf_dim_feedforward,
            activation='gelu',
            batch_first=True,
            dropout=config.get('tf_dropout', 0.1),
        )
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, num_layers=tf_n_layer)
        
        # Action prediction head
        self.pred_action = nn.Linear(tf_n_embd, config['num_actions'])
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean', label_smoothing=config['label_smoothing'])
    
    def _apply_positional_embedding(self, x):
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]
        return x
    
    def _get_causal_mask(self, seq_len):
        """Generate causal attention mask for autoregressive prediction."""
        if self.causal_mask is None or self.causal_mask.size(0) != seq_len:
            mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
            self.causal_mask = mask.to(self.device)
        return self.causal_mask
    
    def _embed_context_dict(self, context_dict):
        """
        Embed a context dictionary containing states, actions, rewards, next_states.
        
        Args:
            context_dict: Dict with keys 'states', 'actions', 'rewards', 'next_states'
                         Each shaped [batch, seq_len, ...]
        
        Returns:
            context_embed: [batch, seq_len, tf_n_embd]
        """
        states = context_dict['states'].to(self.device)
        actions = context_dict['actions'].to(self.device)
        next_states = context_dict['next_states'].to(self.device)
        rewards = context_dict['rewards'].to(self.device)
        
        # Ensure rewards have shape [B, L, 1]
        if rewards.dim() == 2:
            rewards = rearrange(rewards, 'b l -> b l 1')
        
        # Concatenate context
        context, _ = pack([states, actions, rewards, next_states], 'b l *')
        context_embed = self.embed_context(context)
        
        return context_embed
    
    def forward(self, x):
        """
        Training forward pass with compression simulation.
        
        Args:
            x: Dictionary containing:
                - compression_stages: List of context dicts to compress hierarchically
                - uncompressed_context: Recent context dict (not compressed)
                - query_states: Query state to predict action for
                - target_actions: Ground truth actions
                - num_compression_stages: Number of compression cycles
        """
        query_states = x['query_states'].to(self.device)
        target_actions = x['target_actions'].to(self.device)
        num_stages = x['num_compression_stages']
        
        # Hierarchical compression: apply encoder multiple times
        latent_tokens = None
        
        if num_stages > 0:
            compression_stages = x['compression_stages']
            
            for stage_idx in range(num_stages):
                stage_context = compression_stages[stage_idx]
                
                # Embed this stage's context
                context_embed = self._embed_context_dict(stage_context)  # [B, L, D]
                
                # Combine with previous latents if they exist
                if latent_tokens is not None:
                    encoder_input = torch.cat([latent_tokens, context_embed], dim=1)
                else:
                    encoder_input = context_embed
                
                # Compress to latent tokens
                latent_tokens = self.encoder(encoder_input)  # [B, n_latent, D]
        
        # Embed uncompressed recent context
        uncompressed_context = x['uncompressed_context']
        uncompressed_embed = self._embed_context_dict(uncompressed_context)  # [B, L_new, D]
        
        # Embed query state
        query_states_embed = self.embed_query_state(map_dark_states(query_states, self.grid_size).to(torch.long))
        query_states_embed = rearrange(query_states_embed, 'b d -> b 1 d')
        
        # Build decoder input: [latents (if exist)] + [uncompressed] + [query]
        if latent_tokens is not None:
            decoder_input, _ = pack([latent_tokens, uncompressed_embed, query_states_embed], 'b * d')
        else:
            decoder_input, _ = pack([uncompressed_embed, query_states_embed], 'b * d')
        
        # Apply positional embedding and causal masking
        decoder_input = self._apply_positional_embedding(decoder_input)
        seq_len = decoder_input.size(1)
        attn_mask = self._get_causal_mask(seq_len)
        
        # Decoder forward
        transformer_output = self.transformer_decoder(decoder_input, mask=attn_mask)
        
        # Predict action from last token (query state position)
        logits_actions = self.pred_action(transformer_output[:, -1])
        
        # Compute loss
        loss_action = self.loss_fn(logits_actions, target_actions)
        acc_action = (logits_actions.argmax(dim=-1) == target_actions).float().mean()
        
        result = {
            'loss_action': loss_action,
            'acc_action': acc_action
        }
        
        return result
    
    def evaluate_in_context(self, vec_env, eval_timesteps, beam_k=0, sample=True):
        """
        Evaluation with dynamic compression when max sequence length is reached.
        
        Uses hierarchical compression: when decoder input exceeds max_seq_length,
        compress [old_latents + some_new_context] -> new_latents, keeping recent context.
        """
        outputs = {}
        outputs['reward_episode'] = []
        
        reward_episode = np.zeros(vec_env.num_envs)
        
        # Initialize
        query_states = vec_env.reset()
        query_states = torch.tensor(query_states, device=self.device, requires_grad=False, dtype=torch.long)
        query_states = rearrange(query_states, 'e d -> e 1 d')
        query_states_embed = self.embed_query_state(map_dark_states(query_states, self.grid_size))
        
        # Start with no latents, only query
        latent_tokens = None
        context_embed = None
        
        for step in range(eval_timesteps):
            query_states_prev = query_states.clone().detach().to(torch.float)
            
            # Build decoder input
            if latent_tokens is not None and context_embed is not None:
                decoder_input, _ = pack([latent_tokens, context_embed, query_states_embed], 'e * d')
            elif context_embed is not None:
                decoder_input, _ = pack([context_embed, query_states_embed], 'e * d')
            else:
                decoder_input = query_states_embed
            
            # Check if we need to compress
            current_seq_len = decoder_input.size(1)
            if current_seq_len > self.max_seq_length:
                # Compression needed: compress [latents + old_context] -> new_latents
                # Keep only recent context uncompressed
                
                # Calculate how much context to compress
                if latent_tokens is not None:
                    n_latent_tokens = latent_tokens.size(1)
                    n_context_tokens = context_embed.size(1)
                    # Total tokens to compress = all latents + (context - keep_recent)
                    keep_recent = self.max_seq_length - self.n_latent - 1  # -1 for query
                    compress_context_len = n_context_tokens - keep_recent
                    
                    if compress_context_len > 0:
                        # Split context
                        context_to_compress = context_embed[:, :compress_context_len, :]
                        context_to_keep = context_embed[:, compress_context_len:, :]
                        
                        # Prepare encoder input: [old_latents, context_to_compress]
                        encoder_input = torch.cat([latent_tokens, context_to_compress], dim=1)
                    else:
                        # Just compress latents, keep all context
                        encoder_input = latent_tokens
                        context_to_keep = context_embed
                else:
                    # No latents yet, compress oldest context
                    keep_recent = self.max_seq_length - self.n_latent - 1
                    context_to_compress = context_embed[:, :-keep_recent, :]
                    context_to_keep = context_embed[:, -keep_recent:, :]
                    encoder_input = context_to_compress
                
                # Compress
                latent_tokens = self.encoder(encoder_input)
                context_embed = context_to_keep
                
                # Rebuild decoder input
                decoder_input, _ = pack([latent_tokens, context_embed, query_states_embed], 'e * d')
            
            # Apply positional embedding and decode
            decoder_input_pos = self._apply_positional_embedding(decoder_input)
            seq_len = decoder_input_pos.size(1)
            attn_mask = self._get_causal_mask(seq_len)
            output = self.transformer_decoder(decoder_input_pos, mask=attn_mask)
            
            # Predict action
            logits = self.pred_action(output[:, -1])
            
            if sample:
                log_probs = F.log_softmax(logits, dim=-1)
                actions = torch.multinomial(log_probs.exp(), num_samples=1)
                actions = rearrange(actions, 'e 1 -> e')
            else:
                actions = logits.argmax(dim=-1)
            
            # Step environment
            query_states, rewards, dones, infos = vec_env.step(actions.cpu().numpy())
            
            # Prepare transition for context
            actions_onehot = rearrange(actions, 'e -> e 1 1')
            actions_onehot = F.one_hot(actions_onehot, num_classes=self.config['num_actions']).float()
            
            reward_episode += rewards
            rewards_tensor = torch.tensor(rewards, device=self.device, requires_grad=False, dtype=torch.float)
            rewards_tensor = rearrange(rewards_tensor, 'e -> e 1 1')
            
            query_states = torch.tensor(query_states, device=self.device, requires_grad=False, dtype=torch.long)
            query_states = rearrange(query_states, 'e d -> e 1 d')
            
            if dones[0]:
                outputs['reward_episode'].append(reward_episode)
                reward_episode = np.zeros(vec_env.num_envs)
                
                states_next = torch.tensor(np.stack([info['terminal_observation'] for info in infos]),
                                          device=self.device, dtype=torch.float)
                states_next = rearrange(states_next, 'e d -> e 1 d')
            else:
                states_next = query_states.clone().detach().to(torch.float)
            
            # Update query state embedding
            query_states_embed = self.embed_query_state(map_dark_states(query_states, self.grid_size))
            
            # Create new context transition and append
            new_transition, _ = pack([query_states_prev, actions_onehot, rewards_tensor, states_next], 'e i *')
            new_transition_embed = self.embed_context(new_transition)
            
            if context_embed is not None:
                context_embed = torch.cat([context_embed, new_transition_embed], dim=1)
            else:
                context_embed = new_transition_embed
        
        outputs['reward_episode'] = np.stack(outputs['reward_episode'], axis=1)
        
        return outputs
