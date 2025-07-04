import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from config import CONFIG

from src.train_finetune import run_finetune 
from src.train_features import run_feature_pipeline
from src.embedding_pipeline import run_embedding_pipeline
from src.zero_shot import run_zero_shot

def main():
    parser = argparse.ArgumentParser(description="Unified LLM Text Classification Pipeline")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["finetune", "feature", "embedding", "zero"],
        help="Pipeline mode to run"
    )
    # Optional overrides
    parser.add_argument("--model_name", type=str, help="Model name or path override")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--device", type=str, help="Device (cpu or cuda)")
    parser.add_argument("--output_dir", type=str, help="Output directory override")
    parser.add_argument("--openai_api_key", type=str, help="OpenAI API key for zero-shot mode")
    parser.add_argument("--text", type=str, help="Text input for zero-shot classification")

    args = parser.parse_args()

    # Make a mutable copy of config
    config = dict(CONFIG)
    config["finetune"] = dict(config["finetune"])

    # Finetune config overrides
    if args.mode == "finetune":
        if args.model_name:
            config["finetune"]["model"] = args.model_name
        if args.epochs:
            config["finetune"]["num_epochs"] = args.epochs
        if args.batch_size:
            config["finetune"]["batch_size"] = args.batch_size
        if args.learning_rate:
            config["finetune"]["learning_rate"] = args.learning_rate
        if args.device:
            config["finetune"]["device"] = args.device
        if args.output_dir:
            config["finetune"]["output_dir"] = args.output_dir

    # Zero-shot config check
    if args.mode == "zero":
        if not args.text:
            raise ValueError("In zero-shot mode, you must pass input text with --text")
        if args.openai_api_key:
            os.environ["OPENAI_API_KEY"] = args.openai_api_key
        elif not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OpenAI API key required. Set with --openai_api_key or env var OPENAI_API_KEY")
        config["zero_shot"]["api_key_env"] = "OPENAI_API_KEY"

    print(f"[INFO] Running mode: {args.mode}")

    if args.mode == "finetune":
        run_finetune(config)
    elif args.mode == "feature":
        run_feature_pipeline(config)
    elif args.mode == "embedding":
        run_embedding_pipeline(config)
    elif args.mode == "zero":
        run_zero_shot(config, input_text=args.text)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()
