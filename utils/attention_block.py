import torch.nn as nn


class AttentionBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AttentionBlock, self).__init__()

        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, input_dim = x.size()

        # Compute query, key, and value
        q = self.query(x)  # (batch_size, seq_len, hidden_dim)
        k = self.key(x)  # (batch_size, seq_len, hidden_dim)
        v = self.value(x)  # (batch_size, seq_len, hidden_dim)

        # Compute attention scores and apply attention weights to value
        attended_values = F.scaled_dot_product_attention(q, k, v, dropout_p=0.5)

        return attended_values
