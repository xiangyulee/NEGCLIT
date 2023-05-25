import torch
from torch import nn
import math

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerClassifier, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(input_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(input_dim, nhead, nhid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, src):
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.classifier(output[:, -1, :])
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
class PrunedTransformerBase(nn.Module):

    def __init__(self, num_classes,nlayers, nhead=2,nhid=32, cfg=None,input_dim=76, dropout=0.5):
        
        super(TransformerClassifier, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(input_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(input_dim, nhead, nhid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.classifier = nn.Linear(input_dim, num_classes)
        self.head=nn.Linear(input_dim, num_classes)

    def features(self,x,head=False):
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        if head:
            x = self.head(x[:, -1, :])
      
        return x

    def forward(self, x):
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.classifier(x[:, -1, :])
        return x

def Transformer_E(nclasses, depth=20,cfg=None, bias=True):
    """network element for client"""
    return PrunedTransformerBase(nclasses, depth,cfg=cfg)