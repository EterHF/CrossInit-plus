import argparse
import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

from transformers import CLIPTokenizer,CLIPTextModel
from pathlib import Path
from utils import *
import matplotlib.pyplot as plt

@torch.no_grad()
def infer(
    prompt:str,
    n_images:int,
    pretrained_model_name_or_path,
    learned_embed_name_or_path,
    num_inference_steps=50,
    generator=None,
    device='cpu',
):
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer",torch_dtype=torch.float16)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder",torch_dtype=torch.float16).to(device)

    if learned_embed_name_or_path is None:
        
    embeds_dict=torch.load(learned_embed_name_or_path)
    tokens=list(embeds_dict.keys())
    embeds = [embeds_dict[token]for token in tokens]

    tokenizer.add_tokens(tokens)
    text_encoder.resize_token_embeddings(len(tokenizer))
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    for i,token_id in enumerate(token_ids):
        text_encoder.get_input_embeddings().weight.data[token_id] = embeds[i]

    prompt=prompt.format(" ".join(tokens))

    pipe = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            torch_dtype=torch.float16,
        ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # Track the embeddings at each layer
    init_embeddings = embeds # To store the initial embeddings
    learned_embeddings = []  # To store the learned embeddings

    # Hook to capture the output of each block of the text encoder
    def hook_fn(module, input, output):
        learned_embeddings.append(output.detach().cpu())  # Capture learned embedding

    # Register hooks for each block in the text encoder
    for block in text_encoder.encoder.blocks:
        block.attention.self.register_forward_hook(hook_fn)

    images = pipe(
        prompt,
        generator=generator,
        num_images_per_prompt=n_images,
        num_inference_steps=num_inference_steps
    ).images
    return images, init_embeddings, learned_embeddings


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a testing script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        required=False,
        default=50,
    )
    parser.add_argument(
        "--num_images_per_prompt",
        required=False,
        default=16,
        type=int,
    )
    parser.add_argument(
        "--save_dir",
        required=False,
        default=None,
        type=str,
    )
    parser.add_argument(
        "--device",
        required=False,
        default="cuda:0",
        type=str,
    )
    parser.add_argument(
        "--learned_embedding_path",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--prompt",
        required=False,
        type=str,
        default=None
    )
    parser.add_argument(
        "--n_iter",
        required=False,
        type=int,
        default=1
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    
    if args.prompt is not None and args.prompt_file is not None:
        raise ValueError('`--prompt` cannot be used with `--prompt_file`')
    
    if args.prompt is not None:
        args.prompt=[args.prompt]

    if args.prompt_file is not None:
        with open(args.prompt_file,'r') as f:
            args.prompt=f.read().splitlines()

    if args.save_dir is not None:
        Path(args.save_dir).mkdir(parents=True,exist_ok=True)

    return args

def stat(init_embeddings, learned_embeddings):
    # Convert embeddings to numpy arrays for plotting
    init_embeddings = np.array([embedding.mean(dim=1).numpy() for embedding in init_embeddings])  # mean across tokens
    learned_embeddings = np.array([embedding.mean(dim=1).numpy() for embedding in learned_embeddings])  # mean across tokens
    
    # Calculate L2 norm for each layer
    l2_norm_init = np.linalg.norm(init_embeddings, axis=1)
    l2_norm_learned = np.linalg.norm(learned_embeddings, axis=1)
    
    # Calculate cosine similarity between initial and learned embeddings at each block

    def cosine_similarity(A, B):
        dot_product = np.dot(A, B)
        norm_A = np.linalg.norm(A)
        norm_B = np.linalg.norm(B)
        return dot_product / (norm_A * norm_B)
    cosine_sim = cosine_similarity(init_embeddings, learned_embeddings)

    # Plot the L2 norm
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # L2 norm plot (left)
    ax1.plot(l2_norm_init, label="L2 norm of $v_k^{\text{init}}$", color='orange')
    ax1.plot(l2_norm_learned, label="L2 norm of $v_k^{\text{learned}}$", color='blue')
    ax1.set_xlabel("Block")
    ax1.set_ylabel("L2 Norm")
    ax1.set_title("L2 Norm of Initial and Learned Embeddings")
    ax1.legend()

    # Cosine similarity plot (right)
    ax2.plot(cosine_sim, label="Cosine similarity", color='green')
    ax2.set_xlabel("Block")
    ax2.set_ylabel("Cosine Similarity")
    ax2.set_title("Cosine Similarity between Initial and Learned Embeddings")
    ax2.legend()

    # Show the plots
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    args=parse_args()

    generator=None if args.seed is None else torch.Generator(args.device).manual_seed(args.seed)
    save_dir=Path(args.save_dir)

    for prompt in args.prompt:
        for j in range(args.n_iter):
            images, init, learned=infer(
                prompt=prompt,
                n_images=args.num_images_per_prompt,
                pretrained_model_name_or_path=args.pretrained_model_name_or_path,
                learned_embed_name_or_path=args.learned_embedding_path,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
                device=args.device,
            )
            image_save_path=save_dir.joinpath(prompt.replace(' ','_'))
            image_save_path.mkdir(exist_ok=True,parents=True)
            for i,image in enumerate(images):
                image.save(image_save_path/f'{i+j*args.num_images_per_prompt}.jpg')