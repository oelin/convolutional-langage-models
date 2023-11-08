#@markdown

@dataclass
class CausalCNNOptions:
    """Causal CNN options."""

    hidden_channels: int
    kernel_size: int
  

class CausalCNN(nn.Module):
    """Causal CNN."""
    
    def __init__(self, options: CausalCNNOptions) -> None:
        """Causal CNN options."""
        
        super(CausalCNN, self).__init__()
        self.options = options
        
        self.convolution_1 = nn.Conv1d(in_channels=1, out_channels=options.hidden_channels, kernel_size=options.kernel_size, stride=1, padding=0, dilation=1)
        self.convolution_2 = nn.Conv1d(in_channels=options.hidden_channels, out_channels=options.hidden_channels, kernel_size=options.kernel_size, stride=1, padding=0, dilation=2)
        self.convolution_3 = nn.Conv1d(in_channels=options.hidden_channels, out_channels=1, kernel_size=options.kernel_size, stride=1, padding=0, dilation=4)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        
        # input_padding = (self.options.kernel_size, -1)
        # padding = (self.options.kernel_size - 1, 0)
        
        x = F.relu(self.convolution_1(F.pad(x, ((self.options.kernel_size) * 1, -1))))
        x = F.relu(self.convolution_2(F.pad(x, ((self.options.kernel_size - 1) * 2, 0))))
        x = self.convolution_3(F.pad(x, ((self.options.kernel_size - 1) * 4, 0)))
        
        return x
