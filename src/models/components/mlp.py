from typing import List
from torch import nn
from torch.nn import init

ACTIVATIONS_FUNCS = {
    "relu": nn.ReLU(),
    "gelu": nn.GELU(),
    "leaky_relu": nn.LeakyReLU(),
}

class MLPBase(nn.Module):
    def __init__(
        self,
        input_size: int = 768,
        hidden_dim: int = 1024,
        num_layers: int = 2,
        output_size: int = 8,
        dropout: float = 0.1,
        activation_func: str = "relu",
    ) -> None:
        super().__init__()

        # Validate and set the activation function
        activation_func = activation_func.lower()
        if activation_func not in ACTIVATIONS_FUNCS:
            raise ValueError(
                f"Unsupported activation function: {activation_func}. "
                f"Supported activations are: {list(ACTIVATIONS_FUNCS.keys())}"
            )
        self.activation_func = activation_func
        activation_layer = ACTIVATIONS_FUNCS[self.activation_func]

        layers: List[nn.Module] = []
        current_dim = input_size

        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(activation_layer)
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, output_size))
        self.layers = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initializes the weights of linear layers based on the activation function:
        ReLU/GELU: Kaiming (He) initialization with a=0, Leaky ReLU: Kaiming (He) initialization with a=0.01
        Biases are initialized to zeros.
        """
        for module in self.layers:
            if isinstance(module, nn.Linear):
                if self.activation_func in ["relu", "gelu"]:
                    # Kaiming initialization with a=0
                    init.kaiming_uniform_(module.weight, a=0, nonlinearity="relu")
                elif self.activation_func == "leaky_relu":
                    # Kaiming initialization with a=0.01
                    init.kaiming_uniform_(module.weight, a=0.01, nonlinearity="leaky_relu")
                if module.bias is not None:
                    init.zeros_(module.bias)

    def forward(self, x):
        logits = self.layers(x)
        return logits