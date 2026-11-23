class TensorCache:
    """Per-chain cache shared by every tensor of one forward pass.

    Holds the kernel maps / coord maps used by `conv3d`, plus whatever the
    caller stores (core/models/utils.py uses `idx_query`, `idx_query_devox`,
    `weights_devox`).
    """

    def __init__(self):
        self.kmaps = {}
        self.cmaps = {}
        self.fusedmaps = {}
        self.nbmaps = {}
        self.nmaps = {}      # stride -> 该层真实行数（容量模式，供转置卷积取用）


__all__ = ["TensorCache"]
