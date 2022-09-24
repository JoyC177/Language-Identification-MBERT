import torch


class ClassifierHead(torch.nn.Module):
    def __init__(
        self, input_dim, output_dim, hidden_dims=None, activation=torch.nn.ReLU
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = []

        # Save arguments
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.activation = activation

        dims = [input_dim] + hidden_dims + [output_dim]

        # Construct layers
        layers = []
        for i_dim, o_dim in zip(dims, dims[1:]):
            layers.append(torch.nn.Linear(i_dim, o_dim))
            layers.append(self.activation())
        layers = layers[:-1]  # remove last activation
        self.classifier = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.classifier(x)
