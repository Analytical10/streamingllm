import argparse
import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="/data/pretrained_models/Llama-2-7b-hf")
    parser.add_argument("--input", default="data/pg19_20k.txt")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--num-sentences", type=int, default=256)
    parser.add_argument("--sentence-length", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model and tokenizer from {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    print(f"Loading data from {args.input}...")
    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    print("Tokenizing data...")
    tokens = tokenizer.encode(text, return_tensors="pt")
    
    total_tokens_needed = args.num_sentences * args.sentence_length
    if tokens.shape[1] < total_tokens_needed:
        tokens_repeated = tokens.repeat(1, (total_tokens_needed // tokens.shape[1]) + 1)
        tokens = tokens_repeated[:, :total_tokens_needed]
    else:
        tokens = tokens[:, :total_tokens_needed]
    
    # Reshape tokens: [256, 16]
    input_ids = tokens.view(args.num_sentences, args.sentence_length).to(model.device)

    print(f"Running forward pass for batch shape {input_ids.shape} and extracting attentions...")
    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True, use_cache=False)
    
    attentions = outputs.attentions
    
    # Selected layers to plot
    target_layers = [0, 1, 2, 15, 31]
    
    # Verify model has these layers
    num_layers = len(attentions)
    target_layers = [L for L in target_layers if L < num_layers]
    
    print(f"Plotting attention maps for layers: {target_layers}")
    fig, axes = plt.subplots(1, len(target_layers), figsize=(4 * len(target_layers), 4))
    if len(target_layers) == 1:
        axes = [axes]
    
    for ax, layer_idx in zip(axes, target_layers):
        # Attention shape: [batch_size, num_heads, seq_len, seq_len]
        layer_attn = attentions[layer_idx]
        # Average over batch (dim=0) and heads (dim=1) -> [seq_len, seq_len]
        avg_attn = layer_attn.mean(dim=0).mean(dim=0).cpu().to(torch.float32).numpy()
        
        # 使用自定义的 vmax 以提升对比度，防止峰值(如 Sink Token)把其他特征压实; 改用易于区分强度的 'Blues' / 'magma' 热力图颜色
        sns.heatmap(avg_attn, ax=ax, cmap="magma", vmin=0, vmax=0.15, cbar=True)
        ax.set_title(f"Layer {layer_idx} (Avg Heads)")
        ax.set_xlabel("Key Position")
        ax.set_ylabel("Query Position")
    
    plt.tight_layout()
    output_path = os.path.join(args.output_dir, "exp3_attention_maps.png")
    plt.savefig(output_path, dpi=300)
    print(f"Saved attention maps to {output_path}")

if __name__ == "__main__":
    main()
