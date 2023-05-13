import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange


class Generator(nn.Module):
    def __init__(self, seq_len=30, feature_dim=51, latent_dim=51, embed_dim=128, num_heads=4, num_layers=3):
        super(Generator, self).__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.embedding = nn.Linear(latent_dim, seq_len * embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(seq_len, embed_dim))

        self.transformer_layers = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])

        self.decoder = nn.Linear(embed_dim, feature_dim)  # Change here

    def forward(self, z):
        batch_size = z.shape[0]  # Store the batch size
        x = self.embedding(z).view(batch_size, self.seq_len, self.embed_dim)
        x = x + self.positional_encoding
        for layer in self.transformer_layers:
            x = layer(x)
        output = self.decoder(x)
        return output








class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(TransformerEncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.feed_forward = FeedForward(embed_dim)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)




    def forward(self, x):
        residual = x
        x = self.layer_norm1(x)
        x = x + self.attention(x)
        x = x + residual
        residual = x
        x = self.layer_norm2(x)
        x = x + self.feed_forward(x)
        x = x + residual
        return x








class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
       
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)
       
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
       
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
       
        query = rearrange(query, 'b n (h d) -> b h n d', h=self.num_heads)
        key = rearrange(key, 'b n (h d) -> b h n d', h=self.num_heads)
        value = rearrange(value, 'b n (h d) -> b h n d', h=self.num_heads)
       
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / self.head_dim
        attention_probs = F.softmax(attention_scores, dim=-1)
       
        weighted_sum = torch.matmul(attention_probs, value)
        weighted_sum = rearrange(weighted_sum, 'b h n d -> b n (h d)')
       
        x = self.fc(weighted_sum)
        return x
   
class FeedForward(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, dropout=0.1):
        super(FeedForward, self).__init__()
        hidden_dim = embed_dim * expansion_factor
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)




    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
   


class Discriminator(nn.Module):
    def __init__(self, seq_len=30, feature_dim=51, embed_dim=128, num_heads=4, num_layers=3):
        super(Discriminator, self).__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.embedding = nn.Linear(seq_len * feature_dim, seq_len * embed_dim)  # Change here
        self.positional_encoding = nn.Parameter(torch.zeros(seq_len, embed_dim))

        self.transformer_layers = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])

        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, x):
        batch_size = x.shape[0]  # Store the batch size
        x = self.embedding(x.view(batch_size, -1)).view(batch_size, self.seq_len, self.embed_dim)  # Change here
        x = x + self.positional_encoding
        for layer in self.transformer_layers:
            x = layer(x)
        output = self.classifier(x[:, -1, :])  # Use only the last token representation for classification
        return output
