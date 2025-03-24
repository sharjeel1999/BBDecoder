import torch.nn.functional as F

def cosine_similarity(tensor, dim):
    """
    Calculates the similarity of a 4D tensor across dimension 1 using vectorized operations.

    Args:
        tensor: A 4D tensor of shape [batch_size, channels, height, width].
        dim: The dimension to calculate the similarity across.

    Returns:
        A 3D tensor of shape [batch_size, height, width] representing the similarity.
    """
    batch_size, channels, height, width = tensor.shape
    # Reshape tensor to [batch_size, channels, height * width]
    reshaped_tensor = tensor.reshape(batch_size, channels, height * width)

    # Expand dimensions for pairwise comparison
    channel_i = reshaped_tensor.unsqueeze(2)  # [batch_size, channels, 1, height * width]
    channel_j = reshaped_tensor.unsqueeze(1)  # [batch_size, 1, channels, height * width]

    # Calculate cosine similarity for all pairs of channels
    similarity_matrix = F.cosine_similarity(channel_i, channel_j, dim = -1)  # [batch_size, channels, channels]

    # Sum the similarity values, excluding the diagonal (similarity with itself)
    similarity_sum = similarity_matrix.sum(dim = 1) - 1

    # Average the similarity
    num_pairs = channels - 1
    similarity = similarity_sum / num_pairs

    # Reshape to the desired output shape
    similarity = similarity.mean(dim = 1)
    return similarity

def calculate_kl_divergence_torch(tensor, reduction = 'mean'):
    """
    Calculates the KL divergence of a 4D tensor across dimension 1 using vectorized operations.

    Args:
        tensor: A 4D tensor of shape [batch_size, channels, height, width].
        reduction (str, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. Default: 'mean'.

    Returns:
        A 1D tensor of shape [batch_size] representing the overall KL divergence.
    """
    batch_size, channels, height, width = tensor.shape
    # Reshape tensor to [batch_size, channels, height * width]
    reshaped_tensor = tensor.reshape(batch_size, channels, height * width)

    # Ensure non-negativity and create probability distributions.
    p = F.softmax(reshaped_tensor, dim = 2)
    q = F.softmax(reshaped_tensor, dim = 2)

    # Expand dimensions for pairwise comparison
    channel_i = p.unsqueeze(2)  # [batch_size, channels, 1, height * width]
    channel_j = q.unsqueeze(1)  # [batch_size, 1, channels, height * width]

    # Calculate KL divergence for all pairs of channels
    kl_divergence_matrix = F.kl_div(channel_i, channel_j, reduction='none').sum(dim = -1)  # [batch_size, channels, channels]

    # Sum the KL divergence values, excluding the diagonal (divergence with itself)
    kl_divergence_sum = kl_divergence_matrix.sum(dim = 1) - 0  # KL Div with itself is 0

    # Average the KL divergence over spatial dimensions and channel pairs for each batch
    overall_kl_divergence = kl_divergence_sum.mean(dim = 1)

    return overall_kl_divergence
