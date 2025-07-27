import argparse
from data_pipline import process_pdf_pipeline

def cli():
    parser = argparse.ArgumentParser(description="📄 PDF Chunking Pipeline")
    parser.add_argument("url", type=str, help="PDF URL to download and process")
    parser.add_argument("--output", type=str, default="chunks_output.csv", help="Output CSV path")
    parser.add_argument("--save_path", type=str, default="data/downloaded.pdf", help="Path to save downloaded PDF")
    parser.add_argument("--chunk_size", type=int, default=10, help="Number of sentences per chunk")
    parser.add_argument("--plot", action="store_true", help="Plot chunk size histogram")

    args = parser.parse_args()

    process_pdf_pipeline(
        url=args.url,
        save_path=args.save_path,
        chunk_size=args.chunk_size,
        output_csv=args.output,
        show_stats=args.plot
    )

if __name__ == "__main__":
    cli()
    print("✅ daata Pipeline completed.")