import matplotlib.pyplot as plt
import logging

def plot_chunk_stats(chunks, output_img="chunk_stats.png", show=False):
    token_lengths = [chunk["chunk_token_estimate"] for chunk in chunks]
    plt.figure(figsize=(10, 5))
    plt.hist(token_lengths, bins=20, color="skyblue", edgecolor="black")
    plt.title("📊 Chunk Token Estimate Distribution")
    plt.xlabel("Estimated Tokens per Chunk")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_img)
    logging.info(f"📊 Histogram saved to {output_img}")
    if show:
        plt.show()
