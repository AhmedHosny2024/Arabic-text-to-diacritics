from torch import nn
import torch

class LSTM(nn.Module):
    def __init__( self,
        inp_vocab_size: int,targ_vocab_size: int,embedding_dim: int = 512,
        layers_units: list[int] = [256, 256, 256],use_batch_norm: bool = False):
        super().__init__()
        self.target_vocab_size = targ_vocab_size
        self.embedding = nn.Embedding(inp_vocab_size, embedding_dim)
        # create LSTM layers numer = layers_units
        layers_units = [embedding_dim//2] + layers_units
        layers=[]
        for i in range(1,len(layers_units)):
            layers.append(nn.LSTM(layers_units[i-1]*2,layers_units[i],
                                  batch_first=True,bidirectional=True))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(layers_units[i]))
        self.layers=nn.ModuleList(layers)
        self.projection = nn.Linear(layers_units[-1]*2, targ_vocab_size)
        self.layers_units = layers_units
        self.use_batch_norm = use_batch_norm

    def forward(self, text: torch.Tensor):
        output = self.embedding(text)
        for i , layer in enumerate(self.layers):
            if(isinstance(layer,nn.BatchNorm1d)):
                output = output.permute(0,2,1)
                output = layer(output)
                output = output.permute(0,2,1)
                continue
            if i>0:
                output,(hn,cn)=layer(output,(hn,cn))
            else:
                output,(hn,cn)=layer(output)
        output = self.projection(output)
        return output
      

        
    
 