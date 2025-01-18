import argparse
import torch
from transformers import CLIPTokenizer, CLIPTextModel
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

pretrained_model_name_or_path = "stabilityai/stable-diffusion-2-1-base"

def stat(path=None):
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer", torch_dtype=torch.float16)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder", torch_dtype=torch.float16).cuda()

    if path is not None:
        embeds_dict = torch.load(path)
        tokens = list(embeds_dict.keys())
        embeds = [embeds_dict[token] for token in tokens]

        tokenizer.add_tokens(tokens)
        text_encoder.resize_token_embeddings(len(tokenizer))
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        for i, token_id in enumerate(token_ids):
            text_encoder.get_input_embeddings().weight.data[token_id] = embeds[i]
        tokens = ' '.join(tokens)
    else:
        raise ValueError("Path cannot be None.")

    # Track the embeddings at each layer
    embeddings = [torch.stack(embeds)]

    # Hook to capture the output of each block of the text encoder
    def hook_fn(module, input, output):
        embeddings.append(output[0].squeeze(0).detach().cpu())  # Capture learned embedding

    # Register hooks for each block in the text encoder
    for block in text_encoder.text_model.encoder.layers:
        block.self_attn.register_forward_hook(hook_fn)

    input_ids = tokenizer(
        tokens,
        padding="do_not_pad",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids[:, 1:-1]

    _ = text_encoder(input_ids.cuda())

    return embeddings

def draw(init_embeddings, learned_embeddings):
    # Convert embeddings to numpy arrays for plotting
    init_embeddings_np = np.array([embedding.numpy() for embedding in init_embeddings])
    learned_embeddings_np = np.array([embedding.numpy() for embedding in learned_embeddings])

    # Calculate L2 norm for each layer
    l2_norm_init = np.linalg.norm(init_embeddings_np.reshape(init_embeddings_np.shape[0], -1), axis=1)
    l2_norm_learned = np.linalg.norm(learned_embeddings_np.reshape(learned_embeddings_np.shape[0], -1), axis=1)

    # Calculate cosine similarity
    cosine_sim1 = np.array([F.cosine_similarity(init_embeddings[0], learned_embeddings[i]).mean().item() for i in range(len(learned_embeddings))])
    cosine_sim2 = np.array([F.cosine_similarity(init_embeddings[0], init_embeddings[i]).mean().item() for i in range(len(init_embeddings))])

    # Plot the L2 norm
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # L2 norm plot
    ax1.plot(l2_norm_init, label="L2 norm of $v_k^{\\text{init}}$", color='orange')
    ax1.plot(l2_norm_learned, label="L2 norm of $v_k^{\\text{learned}}$", color='blue')
    ax1.set_xlabel("Block")
    ax1.set_ylabel("L2 Norm")
    ax1.set_title("L2 Norm of Initial and Learned Embeddings")
    ax1.legend()

    # Cosine similarity plot
    ax2.plot(cosine_sim1, label="Cosine similarity $v_0^{\\text{init}} v_k^{\\text{learned}}$", color='blue')
    ax2.plot(cosine_sim2, label="Cosine similarity $v_0^{\\text{init}} v_k^{\\text{init}}$", color='orange')
    ax2.set_xlabel("Block")
    ax2.set_ylabel("Cosine Similarity")
    ax2.set_title("Cosine Similarity between Initial and Learned Embeddings")
    ax2.legend()

    # Show the plots
    plt.tight_layout()
    plt.savefig("fig1.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare initial and learned embeddings.")
    parser.add_argument("--init_path", type=str, required=True, help="Path to the initial embedding file.")
    parser.add_argument("--learned_path", type=str, required=True, help="Path to the learned embedding file.")

    args = parser.parse_args()

    init = stat(args.init_path)
    learned = stat(args.learned_path)
    draw(init, learned)