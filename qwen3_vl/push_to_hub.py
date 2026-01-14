# merge_and_push.py
import argparse
import os
from unsloth import FastVisionModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter with base model and push to Hugging Face Hub"
    )
    
    # Model configuration
    parser.add_argument(
        "--lora-model-id",
        type=str,
        required=True,
        help="Hugging Face repo ID for the LoRA model (e.g., ha684/qwen3-vl-2b-lora-finetuned)"
    )
    parser.add_argument(
        "--new-model-id",
        type=str,
        required=True,
        help="Hugging Face repo ID for the merged model (e.g., ha684/qwen3-vl-2b-1.0)"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face API token (if not set, will use HF_TOKEN env variable)"
    )
    
    # Model loading options
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        default=False,
        help="Load model in 4bit (not recommended for merging)"
    )
    parser.add_argument(
        "--load-in-16bit",
        action="store_true",
        default=True,
        help="Load model in 16bit (recommended for merging)"
    )
    
    # Save method
    parser.add_argument(
        "--save-method",
        type=str,
        choices=["merged_16bit", "gguf"],
        default="merged_16bit",
        help="Save method: merged_16bit (for vLLM/SGLang) or gguf (for Ollama/Llama.cpp)"
    )
    
    # GGUF-specific options
    parser.add_argument(
        "--quantization-method",
        type=str,
        default="q4_k_m",
        choices=["q4_k_m", "q8_0", "f16", "q2_k", "q3_k_m", "q5_k_m", "q6_k"],
        help="Quantization method for GGUF (only used if save-method is gguf)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set up Hugging Face token
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError(
            "HF_TOKEN must be provided via --hf-token argument or HF_TOKEN environment variable"
        )
    os.environ["HF_TOKEN"] = hf_token
    
    # Load model and adapter
    print(f"⏳ Loading LoRA model from {args.lora_model_id}...")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=args.lora_model_id,
        load_in_4bit=args.load_in_4bit,
        load_in_16bit=args.load_in_16bit,
    )
    
    # Merge and push based on save method
    if args.save_method == "merged_16bit":
        print(f"🚀 Pushing Merged 16bit model to {args.new_model_id}...")
        model.push_to_hub_merged(
            args.new_model_id,
            tokenizer,
            save_method="merged_16bit",
            token=hf_token
        )
    elif args.save_method == "gguf":
        print(f"🚀 Pushing GGUF model to {args.new_model_id}...")
        print(f"   Quantization method: {args.quantization_method}")
        model.push_to_hub_gguf(
            args.new_model_id,
            tokenizer,
            quantization_method=args.quantization_method,
            token=hf_token
        )
    
    print("✅ DONE! Check your Hugging Face Hub.")


if __name__ == "__main__":
    main()