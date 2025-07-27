import pandas as pd
import logging

def save_chunks_to_csv(chunks, output_path="chunks_output.csv"):
    df = pd.DataFrame(chunks)
    df.to_csv(output_path, index=False)
    logging.info(f"✅ Chunks saved to {output_path}")
