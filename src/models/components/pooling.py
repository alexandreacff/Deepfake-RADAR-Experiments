from torch import nn
import torch
import torch.nn.functional as F

class Pooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError

class AttentiveStatisticsPooling(Pooling):
    """
    AttentiveStatisticsPooling
    Paper: Attentive Statistics Pooling for Deep Speaker Embedding
    https://arxiv.org/pdf/1803.10963.pdf

    Input:
    x: (batch_size, T, feat_dim)
    Output:
    (batch_size, feat_dim*2)
    """
    def __init__(self, input_size):
        super().__init__()
        self._indim = input_size
        self.sap_linear = nn.Linear(input_size, input_size)
        self.attention = nn.Parameter(torch.FloatTensor(input_size, 1))
        torch.nn.init.normal_(self.attention, mean=0, std=1)

    def forward(self, xs):
        """
        Args:
            xs: Input tensor of shape (batch_size, T, feat_dim)
        Returns:
            Pooled features of shape (batch_size, feat_dim*2)
        """
        # Calculate attention weights
        h = torch.tanh(self.sap_linear(xs))
        w = torch.matmul(h, self.attention).squeeze(dim=2)
        w = F.softmax(w, dim=1).unsqueeze(2)

        # Calculate weighted mean
        mu = torch.sum(xs * w, dim=1)
        # Calculate weighted standard deviation
        rh = torch.sqrt((torch.sum((xs**2) * w, dim=1) - mu**2).clamp(min=1e-5))
        # Concatenate mean and standard deviation
        pooled = torch.cat((mu, rh), dim=1)
        return pooled
