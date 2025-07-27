"""
RAG Pipeline CLI Tool

This script lets you run your RAG pipeline in modular stages or as a full pipeline.

Usage Examples:
---------------
# Run just the data pipeline with your PDF URL and save location
python rag_pipeline.py data --url http://example.com/file.pdf --save-path ./assets/raw.pdf --chunk-size 512 --show-stats

# Run embedding only with a custom embedding model key
python rag_pipeline.py embed --model-key sentence-transformers/all-MiniLM-L6-v2

# Run inference only on GPU
python rag_pipeline.py inf --device cuda

# Run the full pipeline (data + embed + inference)
python rag_pipeline.py all \
  --url http://example.com/file.pdf \
  --save-path ./assets/raw.pdf \
  --chunk-size 512 \
  --show-stats \
  --model-key sentence-transformers/all-MiniLM-L6-v2 \
  --device cuda
"""

import argparse
import sys
import os

# Import your modules (adjust these if needed)
from data import data_pipeline
from embedding.embed_and_store import main as embed_main
from inference.inference import main as inf_main

# ----------------------------
# Pipeline Step: Data Pipeline
# ----------------------------
def run_data_pipeline(url, save_path, chunk_size, show_stats):
    """
    Downloads the PDF, extracts metadata and pages,
    splits into chunks, and optionally shows chunk statistics.
    """
    print("[Pipeline] Running Data Pipeline...")
    data_pipeline.process_pdf_pipeline(
        url=url,
        save_path=save_path,
        chunk_size=chunk_size,
        output_csv=None,  # Optional: add output path if you want CSV elsewhere
        show_stats=show_stats,
    )


# ----------------------------
# Pipeline Step: Embedding
# ----------------------------
def run_embedding(model_key):
    """
    Embeds previously saved text chunks using the specified embedding model.
    """
    print(f"[Pipeline] Running Embedding with model: {model_key} ...")
    embed_main(model_key=model_key)


# ----------------------------
# Pipeline Step: Inference
# ----------------------------
def run_inference(device):
    """
    Loads embeddings and runs retrieval + generation on user queries.
    """
    print(f"[Pipeline] Running Inference on device: {device} ...")
    inf_main(device=device)


# ----------------------------
# Argument Parsing & Routing
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Run RAG pipeline stages modularly")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---------- DATA ----------
    data_parser = subparsers.add_parser("data", help="Run data pipeline")
    data_parser.add_argument("--url", type=str, required=True, help="PDF URL to download")
    data_parser.add_argument("--save-path", type=str, required=True, help="Where to save the downloaded PDF")
    data_parser.add_argument("--chunk-size", type=int, default=512, help="Chunk size for splitting (default: 512)")
    data_parser.add_argument("--show-stats", action="store_true", help="Show chunk stats visualization")

    # ---------- EMBED ----------
    embed_parser = subparsers.add_parser("embed", help="Run embedding pipeline")
    embed_parser.add_argument("--model-key", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                              help="Embedding model key (default: sentence-transformers/all-MiniLM-L6-v2)")

    # ---------- INFERENCE ----------
    inf_parser = subparsers.add_parser("inf", help="Run inference pipeline")
    inf_parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                            help="Device to run inference on (default: cpu)")

    # ---------- ALL ----------
    all_parser = subparsers.add_parser("all", help="Run full pipeline sequentially")
    all_parser.add_argument("--url", type=str, required=True, help="PDF URL to download")
    all_parser.add_argument("--save-path", type=str, required=True, help="Where to save the downloaded PDF")
    all_parser.add_argument("--chunk-size", type=int, default=512, help="Chunk size (default: 512)")
    all_parser.add_argument("--show-stats", action="store_true", help="Show chunk stats visualization")
    all_parser.add_argument("--model-key", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                            help="Embedding model key (default: sentence-transformers/all-MiniLM-L6-v2)")
    all_parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                            help="Device for inference (default: cpu)")

    # --------------------
    # Dispatch CLI command
    # --------------------
    args = parser.parse_args()

    if args.command == "data":
        run_data_pipeline(args.url, args.save_path, args.chunk_size, args.show_stats)

    elif args.command == "embed":
        run_embedding(args.model_key)

    elif args.command == "inf":
        run_inference(args.device)

    elif args.command == "all":
        run_data_pipeline(args.url, args.save_path, args.chunk_size, args.show_stats)
        run_embedding(args.model_key)
        run_inference(args.device)

    else:
        parser.print_help()
        sys.exit(1)


# ----------------------------
# Entry Point
# ----------------------------
if __name__ == "__main__":
    main()
