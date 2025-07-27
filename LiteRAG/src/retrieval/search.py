import torch

def compute_similarity(query_embedding: torch.Tensor,
                       embeddings: torch.Tensor,
                       metric: str = 'cosine') -> torch.Tensor:
    """
    Compute similarity scores between query_embedding and embeddings using specified metric.

    Args:
        query_embedding (torch.Tensor): Query embedding of shape (1, embedding_dim)
        embeddings (torch.Tensor): Embeddings matrix of shape (num_embeddings, embedding_dim)
        metric (str): Similarity metric - 'dot', 'cosine', or 'euclidean'

    Returns:
        torch.Tensor: Similarity scores (higher is better)
    """
    if metric == 'dot':
        scores = torch.mm(query_embedding, embeddings.T)[0]
    elif metric == 'cosine':
        scores = torch.nn.functional.cosine_similarity(query_embedding, embeddings)
    elif metric == 'euclidean':
        distances = torch.cdist(query_embedding, embeddings)[0]
        scores = -distances  # invert so that smaller distance => higher score
    else:
        raise ValueError(f"Unknown similarity metric: {metric}")
    return scores
