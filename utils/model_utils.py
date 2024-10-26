import torch
from torch import nn

def loss_edges(y_pred_edges, y_edges, edge_cw):
    """
    Loss function for edge predictions.

    Args:
        y_pred_edges: Predictions for edges (batch_size, num_nodes, num_nodes)
        y_edges: Targets for edges (batch_size, num_nodes, num_nodes)
        edge_cw: Class weights for edges loss

    Returns:
        loss_edges: Value of loss function
    """
    # Ensure data types and contiguity for NLLLoss
    y_pred_edges = y_pred_edges.float().contiguous()  # Convert predictions to Float and make contiguous
    y_edges = y_edges.long().contiguous()             # Convert targets to Long and make contiguous
    edge_cw = edge_cw.float().contiguous() if edge_cw is not None else None  # Convert weights to Float and make contiguous if provided

    # Edge loss calculation
    y = torch.log_softmax(y_pred_edges, dim=-1).contiguous()  # B x V x V x voc_edges
    if y.dim() > 2:
        y = y.permute(0, 3, 1, 2).contiguous()  # B x voc_edges x V x V

    # Compute NLL loss
    loss_edges = nn.NLLLoss(weight=edge_cw)(y, y_edges)
    return loss_edges
