import torch


def tensor_gather(src: torch.Tensor, index: torch.Tensor, dim: int) -> torch.Tensor:
    """Gathers a tensor along a given dimension using index as the gather indices"""
    assert src.ndim >= dim
    assert index.ndim == 1
    assert index.shape[0] == src.shape[dim]

    index_expanded = index.view([1 if i != dim else index.shape[0] for i in range(src.ndim)])
    index_expanded = index_expanded.expand(src.shape)

    gathered_shape = list(src.shape)
    gathered_shape[dim] = int(index.max().item()) + 1
    src_gathered = torch.zeros(gathered_shape, dtype=src.dtype, device=src.device)
    return src_gathered.scatter_add_(dim, index_expanded, src)
