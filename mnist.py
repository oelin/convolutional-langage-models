#@markdown

@dataclass
class CausalCNNOptions:
    """Causal CNN options."""

    hidden_channels: int
    kernel_size: int
    vocabulary_size: int
    sequence_length: int
    embedding_dimension: int


class CausalCNN(nn.Module):
    """Causal CNN."""
    
    def __init__(self, options: CausalCNNOptions) -> None:
        """Initializes the module."""
        
        super(CausalCNN, self).__init__()
        self.options = options
        
        self.position_embedding = nn.Embedding(num_embeddings=options.sequence_length, embedding_dim=options.embedding_dimension)
        
        self.convolution_1 = nn.Conv1d(in_channels=options.embedding_dimension + options.vocabulary_size, out_channels=options.hidden_channels, kernel_size=options.kernel_size, stride=1, padding=0, dilation=1)
        self.convolution_2 = nn.Conv1d(in_channels=options.hidden_channels, out_channels=options.hidden_channels, kernel_size=options.kernel_size, stride=1, padding=0, dilation=2)
        self.convolution_3 = nn.Conv1d(in_channels=options.hidden_channels, out_channels=options.hidden_channels, kernel_size=options.kernel_size, stride=1, padding=0, dilation=4)
        self.convolution_4 = nn.Conv1d(in_channels=options.hidden_channels, out_channels=options.hidden_channels, kernel_size=options.kernel_size, stride=1, padding=0, dilation=16)
        self.convolution_5 = nn.Conv1d(in_channels=options.hidden_channels, out_channels=options.hidden_channels, kernel_size=options.kernel_size, stride=1, padding=0, dilation=64)
        self.convolution_6 = nn.Conv1d(in_channels=options.hidden_channels, out_channels=options.vocabulary_size, kernel_size=options.kernel_size, stride=1, padding=0, dilation=128)

        self.padding_1 = (self.options.kernel_size, -1)
        self.padding_2 = ((self.options.kernel_size - 1) * 2, 0, 0, 0)
        self.padding_3 = ((self.options.kernel_size - 1) * 4, 0, 0, 0)
        self.padding_4 = ((self.options.kernel_size - 1) * 16, 0, 0, 0)
        self.padding_5 = ((self.options.kernel_size - 1) * 64, 0, 0, 0)
        self.padding_6 = ((self.options.kernel_size - 1) * 128, 0, 0, 0)
        
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        
        # Embeddings.
 
        tokens = tokens.to(int).cuda()
        positions = torch.arange(self.options.sequence_length).to(int).cuda()
        positions = positions.view(1, -1).repeat(tokens.size(0), 1)
        
        token_embeddings = F.one_hot(tokens, num_classes=self.options.vocabulary_size)
        position_embeddings = self.position_embedding(positions)
        
        x = torch.cat((token_embeddings, position_embeddings), dim=-1)
        
        # CNN.
        
        x = x.transpose(-2, -1)
        x = F.leaky_relu(self.convolution_1(F.pad(x, self.padding_1)))
        x = F.leaky_relu(self.convolution_2(F.pad(x, self.padding_2)))
        x = F.leaky_relu(self.convolution_3(F.pad(x, self.padding_3)))
        x = F.leaky_relu(self.convolution_4(F.pad(x, self.padding_4)))
        x = F.leaky_relu(self.convolution_5(F.pad(x, self.padding_5)))
        x = self.convolution_6(F.pad(x, self.padding_6))
        x = x.transpose(-2, -1)
        
        # Logits.
        
        x = F.log_softmax(x, dim=-1)
    
        return x
