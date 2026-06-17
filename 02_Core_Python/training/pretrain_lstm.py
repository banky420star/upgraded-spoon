"""
LSTM Autoencoder Pretraining Module

Trains the PPO feature extractor's LSTM as an autoencoder on raw market
observations BEFORE PPO training starts. This gives the LSTM useful
temporal representations (bar patterns, volatility clusters, etc.) from
the start, rather than starting from random weights.

Two modes:
  1. AE (autoencoder): reconstruct the input sequence -> forces LSTM to
     learn meaningful bar-level representations
  2. Next-bar prediction: predict next timestep's features -> forces LSTM
     to learn predictive temporal patterns

Usage:
  from training.pretrain_lstm import pretrain_feature_extractor
  pretrain_feature_extractor(model, env, steps=500, lr=1e-3, mode='ae')
"""

import logging
import os

import numpy as np
import torch

logger = logging.getLogger("pretrain_lstm")


def pretrain_feature_extractor(
    model,
    env,
    steps: int = 200,
    lr: float = 1e-3,
    mode: str = "ae",
    logger_obj=None,
) -> bool:
    """
    Pretrain the feature extractor (LSTM) using reconstruction objective.

    Warms up the LSTM encoder with useful bar representations before PPO
    training starts. Works by adding a lightweight decoder head and training
    the encoder+decoder to reconstruct (or predict) the input sequence.

    Args:
        model: SB3 PPO model (must have model.policy.features_extractor)
        env: VecNormalize wrapped environment
        steps: Number of pretraining gradient steps
        lr: Learning rate for pretraining
        mode: 'ae' (autoencoder) or 'next_bar' (next-bar prediction)
        logger_obj: Optional logger

    Returns:
        True if pretraining ran, False if skipped
    """
    log = logger_obj or logger

    try:
        fe = model.policy.features_extractor
    except AttributeError:
        log.warning("No features_extractor on model policy, skipping LSTM pretrain")
        return False

    if not hasattr(fe, "encoder") or not isinstance(fe.encoder, torch.nn.LSTM):
        log.warning("Feature extractor has no LSTM encoder, skipping LSTM pretrain")
        return False

    encoder = fe.encoder
    hidden_dim = encoder.hidden_size * (2 if encoder.bidirectional else 1)
    seq_len = fe.seq_window if hasattr(fe, "seq_window") else 100
    portfolio_dim = fe.portfolio_dim if hasattr(fe, "portfolio_dim") else 0
    regime_dim = fe.regime_dim if hasattr(fe, "regime_dim") else 0

    # Calculate input dim per bar (what the LSTM actually sees)
    # The LSTM input is: seq_feature_dim + regime_dim
    lstm_input_dim = encoder.input_size

    # Create lightweight decoder that mirrors the encoder
    decoder_lstm = torch.nn.LSTM(
        input_size=hidden_dim,
        hidden_size=hidden_dim // 2,
        num_layers=1,
        batch_first=True,
        bidirectional=False,
    )
    decoder_proj = torch.nn.Linear(hidden_dim // 2, lstm_input_dim)

    decoder_lstm.to(next(encoder.parameters()).device)
    decoder_proj.to(next(encoder.parameters()).device)

    # CRITICAL: include encoder AND decoder in optimizer so LSTM weights actually update
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder_lstm.parameters()) + list(decoder_proj.parameters()),
        lr=lr,
    )
    loss_fn = torch.nn.MSELoss()

    # Collect observations by stepping through env with neutral actions
    all_obs = []
    action_dim = env.action_space.shape[0] if hasattr(env.action_space, "shape") else 6
    neutral_action = np.zeros(action_dim, dtype=np.float32)

    obs = env.reset()
    for step_idx in range(max(steps * 2, 200)):
        all_obs.append(torch.from_numpy(obs.copy()).float())
        obs, _, done, _ = env.step(neutral_action)
        if done:
            obs = env.reset()

    if len(all_obs) < 10:
        log.warning(f"Too few observations for LSTM pretrain ({len(all_obs)}), skipping")
        return False

    obs_tensor = torch.stack(all_obs).to(next(encoder.parameters()).device)
    log.info(f"LSTM pretrain: collected {len(obs_tensor)} obs, starting {steps} gradient steps...")

    # Save original env state and restore after pretraining
    # (pretraining advances the env, so we reset before PPO starts)
    # The env reset at the start of model.learn() will handle this

    # Training loop
    encoder.train(True)
    decoder_lstm.train()
    decoder_proj.train()

    n_batches = max(1, min(steps, len(obs_tensor) // 16))
    batch_size = max(16, len(obs_tensor) // n_batches)
    initial_loss = None
    final_loss = None

    for batch_idx in range(n_batches):
        batch = obs_tensor[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        if len(batch) < 2:
            continue

        # Forward through extractor up to LSTM output (before attention/projection)
        if portfolio_dim:
            seq_features = batch[:, :-portfolio_dim]
            tail = batch[:, -portfolio_dim:]
        else:
            seq_features = batch
            tail = None

        seq = seq_features.view(batch.shape[0], seq_len, -1)

        # Re-inject regime into sequence (mirroring extractor.forward())
        if regime_dim > 0 and tail is not None and tail.shape[-1] >= regime_dim:
            regime = tail[:, -regime_dim:]
            regime_expanded = regime.unsqueeze(1).expand(-1, seq_len, -1)
            seq = torch.cat([seq, regime_expanded], dim=-1)

        # LSTM encoder forward
        encoded_seq, _ = encoder(seq)  # [batch, seq_len, hidden_dim]

        if mode == "ae":
            # Reconstruct input sequence
            decoded, _ = decoder_lstm(encoded_seq)
            reconstructed = decoder_proj(decoded)
            loss = loss_fn(reconstructed, seq)
        elif mode == "next_bar":
            # Predict next bar's features
            decoded, _ = decoder_lstm(encoded_seq[:, :-1, :])
            predicted = decoder_proj(decoded)
            target = seq[:, 1:, :]
            loss = loss_fn(predicted, target)
        else:
            # Default: autoencoder
            decoded, _ = decoder_lstm(encoded_seq)
            reconstructed = decoder_proj(decoded)
            loss = loss_fn(reconstructed, seq)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(decoder_lstm.parameters()) + list(decoder_proj.parameters()),
            1.0,
        )
        optimizer.step()

        if initial_loss is None:
            initial_loss = loss.item()
        final_loss = loss.item()

    loss_reduction = ((initial_loss - final_loss) / (initial_loss + 1e-8)) * 100 if initial_loss else 0
    log.info(
        f"LSTM pretrain done: {n_batches} batches, "
        f"loss {initial_loss:.6f} -> {final_loss:.6f} ({loss_reduction:+.1f}%)"
    )

    # Clean up decoder (no longer needed)
    del decoder_lstm, decoder_proj

    return True
