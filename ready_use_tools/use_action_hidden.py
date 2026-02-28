"""
Example script demonstrating how to use the action hidden states interface
to concatenate action hidden states from different sources.
"""

import torch
from wan_va.modules.model_with_action_hidden import WanTransformer3DModel
from easydict import EasyDict


def example_concatenate_action_hiddens():
    """
    Example: Concatenate action hidden states from two different sources.
    """

    # Initialize the model (use your actual configuration)
    model_config = EasyDict(
        patch_size=[1, 2, 2],
        num_attention_heads=24,
        attention_head_dim=128,
        in_channels=48,
        out_channels=48,
        action_dim=30,
        text_dim=4096,
        freq_dim=256,
        ffn_dim=14336,
        num_layers=30,
        cross_attn_norm=True,
        eps=1e-06,
        rope_max_seq_len=1024,
        pos_embed_seq_len=None,
        attn_mode="torch"
    )

    model = WanTransformer3DModel(**model_config)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Example: Create action tensors from two different sources
    batch_size = 1
    action_dim = 30
    num_frames = 2
    action_per_frame = 16

    # Source 1: First set of actions (e.g., from experiment A)
    actions_1 = torch.randn(batch_size, action_dim, num_frames, action_per_frame, 1).to(device)

    # Source 2: Second set of actions (e.g., from experiment B)
    actions_2 = torch.randn(batch_size, action_dim, num_frames, action_per_frame, 1).to(device)

    # Method 1: Get hidden states for each action set and concatenate
    hidden_1 = model.get_action_hidden_states(actions_1)  # Shape: [B, L1, C]
    hidden_2 = model.get_action_hidden_states(actions_2)  # Shape: [B, L2, C]

    print(f"Hidden states 1 shape: {hidden_1.shape}")
    print(f"Hidden states 2 shape: {hidden_2.shape}")

    # Concatenate the hidden states
    concatenated_hidden = model.concat_action_hidden_states([hidden_1, hidden_2])
    print(f"Concatenated hidden states shape: {concatenated_hidden.shape}")

    # Method 2: Use the concatenated hidden states in forward pass
    input_dict = {
        'noisy_latents': torch.randn(1, action_dim, num_frames * 2, action_per_frame, 1).to(device),
        'timesteps': torch.ones([1], dtype=torch.float32, device=device) * 100,
        'grid_id': torch.arange(32).reshape(1, 32, 1).to(device),
        'text_emb': torch.randn(1, 512, 3072).to(device),
    }

    with torch.no_grad():
        # Pass the concatenated hidden states directly
        output = model(
            input_dict=input_dict,
            action_mode=True,
            action_hidden_states=concatenated_hidden
        )

        print(f"Output shape: {output.shape}")

    # Method 3: Load pre-saved hidden states and concatenate
    # If you have saved hidden states from previous runs:
    # hidden_saved = torch.load('saved_hidden_states.pt')
    # concatenated_hidden = model.concat_action_hidden_states([hidden_saved, hidden_1])


def example_use_saved_action_hidden():
    """
    Example: Load pre-computed action hidden states and use them in inference.
    """

    # This assumes you have previously saved hidden states
    # For example, during inference you might save them like this:
    # action_hidden = model.get_action_hidden_states(actions)
    # torch.save(action_hidden, 'action_hidden_experiment_1.pt')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load saved hidden states
    hidden_states_1 = torch.load('action_hidden_experiment_1.pt', map_location=device)
    hidden_states_2 = torch.load('action_hidden_experiment_2.pt', map_location=device)

    # Initialize model
    model_config = EasyDict(
        patch_size=[1, 2, 2],
        num_attention_heads=24,
        attention_head_dim=128,
        in_channels=48,
        out_channels=48,
        action_dim=30,
        text_dim=4096,
        freq_dim=256,
        ffn_dim=14336,
        num_layers=30,
        cross_attn_norm=True,
        eps=1e-06,
        rope_max_seq_len=1024,
        pos_embed_seq_len=None,
        attn_mode="torch"
    )

    model = WanTransformer3DModel(**model_config)
    model.eval()
    model.to(device)

    # Prepare input (this would come from your actual inference setup)
    input_dict = {
        'noisy_latents': torch.randn(1, 30, 4, 16, 1).to(device),
        'timesteps': torch.ones([1], dtype=torch.float32, device=device) * 100,
        'grid_id': torch.arange(64).reshape(1, 64, 1).to(device),
        'text_emb': torch.randn(1, 512, 3072).to(device),
    }

    # Concatenate hidden states
    concatenated_hidden = torch.cat([hidden_states_1, hidden_states_2], dim=1)

    with torch.no_grad():
        # Use the concatenated hidden states in forward pass
        output = model(
            input_dict=input_dict,
            action_mode=True,
            action_hidden_states=concatenated_hidden
        )
        print(f"Output with concatenated hidden states: {output.shape}")


def example_weighted_concatenation():
    """
    Example: Weighted concatenation of action hidden states.
    Useful for blending actions from different policies.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model_config = EasyDict(
        patch_size=[1, 2, 2],
        num_attention_heads=24,
        attention_head_dim=128,
        in_channels=48,
        out_channels=48,
        action_dim=30,
        text_dim=4096,
        freq_dim=256,
        ffn_dim=14336,
        num_layers=30,
        cross_attn_norm=True,
        eps=1e-06,
        rope_max_seq_len=1024,
        pos_embed_seq_len=None,
        attn_mode="torch"
    )

    model = WanTransformer3DModel(**model_config)
    model.eval()
    model.to(device)

    # Create action tensors from different sources
    actions_policy_a = torch.randn(1, 30, 2, 16, 1).to(device)
    actions_policy_b = torch.randn(1, 30, 2, 16, 1).to(device)

    # Get hidden states
    hidden_a = model.get_action_hidden_states(actions_policy_a)
    hidden_b = model.get_action_hidden_states(actions_policy_b)

    # Apply weights (e.g., 70% policy A, 30% policy B)
    weight_a = 0.7
    weight_b = 0.3

    # Weighted combination
    weighted_hidden_a = hidden_a * weight_a
    weighted_hidden_b = hidden_b * weight_b

    # Concatenate weighted hidden states
    concatenated_hidden = torch.cat([weighted_hidden_a, weighted_hidden_b], dim=1)

    print(f"Policy A contribution: {weight_a * 100}%")
    print(f"Policy B contribution: {weight_b * 100}%")
    print(f"Concatenated hidden states shape: {concatenated_hidden.shape}")

    # Use in forward pass
    input_dict = {
        'noisy_latents': torch.randn(1, 30, 4, 16, 1).to(device),
        'timesteps': torch.ones([1], dtype=torch.float32, device=device) * 100,
        'grid_id': torch.arange(64).reshape(1, 64, 1).to(device),
        'text_emb': torch.randn(1, 512, 3072).to(device),
    }

    with torch.no_grad():
        output = model(
            input_dict=input_dict,
            action_mode=True,
            action_hidden_states=concatenated_hidden
        )
        print(f"Output with weighted concatenation: {output.shape}")


if __name__ == "__main__":
    print("=" * 80)
    print("Example 1: Basic concatenation of action hidden states")
    print("=" * 80)
    example_concatenate_action_hiddens()

    print("\n" + "=" * 80)
    print("Example 2: Using saved action hidden states")
    print("=" * 80)
    # example_use_saved_action_hidden()  # Uncomment if you have saved files

    print("\n" + "=" * 80)
    print("Example 3: Weighted concatenation")
    print("=" * 80)
    example_weighted_concatenation()
