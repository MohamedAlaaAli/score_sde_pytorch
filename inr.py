import torch
import torch.nn as nn

# Hash Encoding for INR
class HashEncoding(nn.Module):
    def __init__(self, num_levels=12, max_entries=2**23, feature_dims=2, coarsest_res=2, finest_res=256):
        super(HashEncoding, self).__init__()
        self.num_levels = num_levels
        self.max_entries = max_entries
        self.feature_dims = feature_dims
        self.coarsest_res = coarsest_res
        self.finest_res = finest_res
        
        # Compute resolution levels (geometric progression from coarsest to finest)
        self.resolutions = torch.logspace(
            start=torch.log2(torch.tensor(coarsest_res, dtype=torch.float32)),
            end=torch.log2(torch.tensor(finest_res, dtype=torch.float32)),
            steps=num_levels,
            base=2.0
        ).long()
        
        # Initialize hash tables: num_levels tables, each with max_entries rows and feature_dims columns
        self.hash_tables = nn.ParameterList([
            nn.Parameter(torch.randn(max_entries, feature_dims) * 0.001)
            for _ in range(num_levels)
        ])
        
        # Precompute hash function coefficients (for simplicity, use prime numbers)
        self.prime1, self.prime2 = 1, 2654435761  # Large primes for hashing

    def hash_function(self, coords, level):
        """Compute hash indices for coordinates at a given resolution level."""
        res = self.resolutions[level]
        # Scale coordinates to resolution grid
        scaled_coords = coords * res
        # Integer grid coordinates
        grid_coords = torch.floor(scaled_coords).long()
        # Compute hash index: (x * prime1 + y * prime2) % max_entries
        hash_indices = (grid_coords[..., 0] * self.prime1 + grid_coords[..., 1] * self.prime2) % self.max_entries
        return hash_indices

    def forward(self, coords):
        """
        Encode 2D coordinates (batch, num_points, 2) into feature vectors.
        Returns: (batch, num_points, num_levels * feature_dims)
        """
        assert coords.dim() == 3 and coords.shape[-1] == 2, f"Expected shape (batch, num_points, 2), got {coords.shape}"
        batch_size, num_points, _ = coords.shape
        features = []
        
        for level in range(self.num_levels):
            # Get hash indices for this level
            indices = self.hash_function(coords, level)
            # Retrieve features from hash table
            level_features = self.hash_tables[level][indices]  # Shape: (batch, num_points, feature_dims)
            features.append(level_features)
        
        # Concatenate features across all levels
        features = torch.cat(features, dim=-1)  # Shape: (batch, num_points, num_levels * feature_dims)
        return features

# MLP for INR
class INRMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_hidden_layers=3, output_dim=1):
        super(INRMLP, self).__init__()
        self.input_dim = input_dim  # num_levels * feature_dims from hash encoding
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.output_dim = output_dim
        
        # Build layers
        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        # Linear activation for output (as specified)
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        Map encoded features to image intensities.
        Input: x (batch, num_points, input_dim)
        Output: (batch, num_points, output_dim)
        """
        return self.mlp(x)

# Combined INR Model (Hash Encoding + MLP)
class INR(nn.Module):
    def __init__(self, num_levels=12, max_entries=2**23, feature_dims=2, coarsest_res=2, 
                 finest_res=256, hidden_dim=128, num_hidden_layers=3, output_dim=1):
        super(INR, self).__init__()
        self.hash_encoding = HashEncoding(
            num_levels=num_levels,
            max_entries=max_entries,
            feature_dims=feature_dims,
            coarsest_res=coarsest_res,
            finest_res=finest_res
        )
        self.mlp = INRMLP(
            input_dim=num_levels * feature_dims,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            output_dim=output_dim
        )

    def forward(self, coords):
        """
        Map 2D coordinates to image intensities.
        Input: coords (batch, num_points, 2)
        Output: intensities (batch, num_points, 1)
        """
        features = self.hash_encoding(coords)  # (batch, num_points, num_levels * feature_dims)
        intensities = self.mlp(features)       # (batch, num_points, output_dim)
        return intensities
