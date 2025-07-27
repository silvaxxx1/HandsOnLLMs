import textwrap

def pretty_print_text(text: str, width=80, max_chars=500):
    """
    Pretty print text with wrapping and truncation for CLI display.
    """
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
    wrapped = textwrap.fill(text, width=width)
    print(wrapped)

def print_top_results(query: str, scores, indices, chunks_metadata, threshold=0.3):
    """
    Print the top results with scores above a confidence threshold.

    Args:
        query (str): The search query.
        scores (list or iterable): Similarity scores.
        indices (list or iterable): Indices of the top scoring chunks.
        chunks_metadata (list of dict): Metadata and chunk text for each embedding.
        threshold (float): Minimum score to print a result.
    """
    print(f"\nQuery: {query}\n")
    print("Top results:")
    found_any = False
    for score, idx in zip(scores, indices):
        if score < threshold:
            continue
        found_any = True
        print(f"Score: {score:.4f}")
        pretty_print_text(chunks_metadata[idx].get("chunk_text", "No text available"))
        print(f"Page number: {chunks_metadata[idx].get('page_number', 'N/A')}")
        print("-" * 40)
    if not found_any:
        print("No results above the confidence threshold.")
