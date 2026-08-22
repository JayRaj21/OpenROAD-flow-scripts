"""
GNN model for macro floorplan prediction.

Input:  Netlist graph with node features (cells/macros) and net edges
Output: (x, y) normalised placement coordinates for each macro node

Architecture: 3-layer GraphSAGE message passing -> per-macro MLP head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import SAGEConv, global_mean_pool

    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class FloorplanGNN(nn.Module):
    """
    Predicts normalised (x, y) macro placement from the netlist graph.

    Args:
        node_feat_dim:  Number of input node features (default 6)
        hidden_dim:     Hidden dimension for message passing layers
        num_layers:     Number of GraphSAGE layers
    """

    def __init__(
        self, node_feat_dim: int = 6, hidden_dim: int = 128, num_layers: int = 3
    ):
        super().__init__()
        if not HAS_PYG:
            raise ImportError(
                "torch_geometric is required. Install with: "
                "pip install torch-geometric"
            )

        self.input_proj = nn.Linear(node_feat_dim, hidden_dim)

        self.convs = nn.ModuleList(
            [SAGEConv(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )

        # Global context: graph-level embedding added to each macro node
        self.global_proj = nn.Linear(hidden_dim, hidden_dim)

        # Per-macro MLP: predicts (x, y) in [0, 1]
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),
            nn.Sigmoid(),
        )

    def forward(self, x, edge_index, batch, macro_mask):
        """
        Args:
            x:           (N, node_feat_dim) node feature matrix
            edge_index:  (2, E) edge connectivity
            batch:       (N,) batch assignment vector
            macro_mask:  (N,) boolean mask identifying macro nodes

        Returns:
            coords: (M, 2) predicted (x, y) for each macro node, values in [0, 1]
        """
        h = F.relu(self.input_proj(x))

        for conv, norm in zip(self.convs, self.norms):
            h = F.relu(norm(conv(h, edge_index)))

        # Graph-level context embedding
        g = global_mean_pool(h, batch)  # (B, hidden)
        g_expanded = g[batch]  # (N, hidden)
        g_proj = F.relu(self.global_proj(g_expanded))

        # Only predict positions for macro nodes
        h_macro = h[macro_mask]  # (M, hidden)
        g_macro = g_proj[macro_mask]  # (M, hidden)

        macro_input = torch.cat([h_macro, g_macro], dim=1)  # (M, hidden*2)
        coords = self.head(macro_input)  # (M, 2)
        return coords


if __name__ == "__main__":
    if not HAS_PYG:
        print("torch_geometric not installed — skipping test")
    else:
        from torch_geometric.data import Data

        model = FloorplanGNN()
        x = torch.randn(50, 6)
        edge_index = torch.randint(0, 50, (2, 200))
        batch = torch.zeros(50, dtype=torch.long)
        macro_mask = torch.zeros(50, dtype=torch.bool)
        macro_mask[:5] = True
        out = model(x, edge_index, batch, macro_mask)
        print(f"Output shape: {out.shape}")
        total = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {total:,}")
