# prompt_builder.py

def build_prompt(contexts: list[str], query: str, style: str = "qa") -> str:
    """
    Builds a prompt for inference using provided context chunks and user query.

    Args:
        contexts (list of str): Retrieved text chunks.
        query (str): The user's question or prompt.
        style (str): Prompt style - can be 'qa', 'summarize', etc.

    Returns:
        str: The constructed prompt ready for generation.
    """
    if style == "qa":
        prompt = """You are a helpful assistant. Use the following context to answer the question.\n\n"""
        for i, chunk in enumerate(contexts):
            prompt += f"Context [{i+1}]: {chunk}\n"
        prompt += f"\nQuestion: {query}\nAnswer:"
    elif style == "summarize":
        prompt = """You are a summarization assistant. Summarize the following documents.\n\n"""
        for i, chunk in enumerate(contexts):
            prompt += f"Document [{i+1}]: {chunk}\n"
        prompt += "\nSummary:"
    else:
        raise ValueError(f"Unsupported prompt style: {style}")

    return prompt


if __name__ == "__main__":
    # Example usage
    sample_contexts = ["The cat is on the roof.", "Cats love climbing and sunbathing."]
    query = "Why is the cat on the roof?"
    prompt = build_prompt(sample_contexts, query)
    print(prompt)
