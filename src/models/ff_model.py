import torch
from torch import nn

from src.preprocessing.dictionary import PADDING

class FFN(torch.nn.Module):
    """
    A custom model for German -> English Text Translation.

    Args:
        source_vocab_size (int): Size of the source vocabulary.
        target_vocab_size (int): Size of the target vocabulary.
        embedding_dim (int): Dimensionality of the embeddings. Defaults to 2048.
        linSource_dim (int): Dimensionality of the source linear layer. Defaults to 1024.
        linTarget_dim (int): Dimensionality of the target linear layer. Defaults to 1024.
        lin1_dim (int): Dimensionality of the first linear layer. Defaults to 1024.
        window_size (int): Size of the context window. Defaults to 2.
        activation (str): Activation function to use. Supported options: 'gelu', 'relu', 'tanh', 'sigmoid'.
            Defaults to 'gelu'.
    """
    def __init__(self, 
                 source_vocab_size, 
                 target_vocab_size, 
                 embedding_dim = 2048, 
                 linSource_dim = 1024, 
                 linTarget_dim = 1024, 
                 lin1_dim = 1024, 
                 window_size = 2,
                 activation = "gelu",
                 strategy = "uniform",
                 zero_bias = True,
                 config = None):
        super().__init__()
        # store config
        self.config = config

        # Store initialization parameters
        self.strategy = strategy
        self.zero_bias = zero_bias
        # Embedding Layer
        # input : [B,2W+1],[B,W]
        # Input: Batch size * (2*window+1), Batch size * window
        self.GERembeddings = nn.Embedding(source_vocab_size, embedding_dim, padding_idx=PADDING)
        self.ENembeddings = nn.Embedding(target_vocab_size, embedding_dim, padding_idx=PADDING)
        # Output dim = Batch size * (2*window+1) * embedding dim
        # Output dim = Batch size * window * embedding dim
        if activation == "gelu":
            # x * F(x) where F(x) is the cumulative distribution function of a Gaussian distribution ;)
            self.activation = nn.GELU()
        elif activation == "relu":
            # max(0, x)
            self.activation = nn.ReLU()
        elif activation == "tanh":
            # tanh(x)
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            # 1 / (1 + exp(-x))
            self.activation = nn.Sigmoid()
        else:
            raise ValueError("Activation function not supported")

        # Input dim = Batch size * (2*window+1) * embedding dim
        # Input dim = Batch size * window * embedding dim
        self.linSource = nn.Linear((2*window_size+1)*embedding_dim, linSource_dim)
        self.batchnormSource = nn.BatchNorm1d(linSource_dim)
        self.linTarget = nn.Linear(window_size*embedding_dim, linTarget_dim)
        self.batchnormTarget = nn.BatchNorm1d(linTarget_dim)
        # Output dim = Batch size * linSource_dim
        # Output dim = Batch size * linTarget_dim
        # Input dim = Batch size * (linSource_dim + linTarget_dim)
        self.lin1 = nn.Linear(linSource_dim + linTarget_dim, lin1_dim)
        # Output dim = Batch size * lin1_dim
        self.batchnormLin1 = nn.BatchNorm1d(lin1_dim)
        # Input dim = Batch size * lin1_dim
        self.lin2 = nn.Linear(lin1_dim, target_vocab_size)
        # Output dim = Batch size * vocab size
        self.softmax = nn.LogSoftmax(dim=1)
        # Output dim = Batch size * vocab size

        # Force uniform if invalid initilaization strategy is given
        if strategy in ["normal", "uniform"]:
            self.apply(self._init_weights)
        

    def _init_weights(self, module):
        """
        Initializes the weights of the given module.

        Args:
            module (torch.nn.Module): The module to initialize.
        """
        if isinstance(module, nn.Linear):
            if self.strategy == "normal":
                module.weight.data.normal_(mean=0.0, std=1.0)
                module.bias.data.normal_(mean=0.0, std=1.0)
            elif self.strategy == "uniform":
                module.weight.data.uniform_(-1.0, 1.0)
                module.bias.data.uniform_(-1.0, 1.0)
            
            if self.zero_bias:
                module.bias.data.zero_()
        
        if isinstance(module, nn.Embedding):
            if self.strategy == "normal":
                module.weight.data.normal_(mean=0.0, std=1.0)
            elif self.strategy == "uniform":
                module.weight.data.uniform_(-1.0, 1.0)



    def forward(self, s, t):
        """
        Forward pass of the model.

        Args:
            s (torch.Tensor): Input tensor representing the source data.
            t (torch.Tensor): Input tensor representing the target data.

        Returns:
            torch.Tensor: Output tensor of the forward pass.
        """
        batch_size = s.size(0)
        # Embedding layer
        x = self.GERembeddings(s)
        y = self.ENembeddings(t)
        # Fully connected Source
        x= x.view(batch_size,-1)
        # Insert Batch Norm between linear layer and activation function as recommended in the paper
        x = self.activation(self.batchnormSource(self.linSource(x)))
        # Fully connected Target 
        y = y.view(batch_size,-1)
        y = self.activation(self.batchnormTarget(self.linTarget(y)))
        # Concatentation Layer
        z = torch.cat((x,y),dim=1)
        # First fully connected layer
        z = self.activation(self.batchnormLin1(self.lin1(z)))
        # Second fully connected layer
        z = self.lin2(z)
        return z
    