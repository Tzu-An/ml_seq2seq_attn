"""Attention mechanism"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DotProdAttention(nn.Module):
    """Basic Dot-Production Attention"""
    def __init__(self):
        super().__init__()

    def forward(self, output, context):
        """Basic Dot-Production Method
            1. compute e = q * k
            2. compute tanh(softmax(e) * k)

            Args:
                output (batch, 1, hidden): output from decoder rnn
                context (batch, seq, hidden): output from encoder rnn
            Returns:
                output (batch, 1, hidden): modified output
                attn (batch, 1, seq): attention state in this step
        """
        attn = torch.bmm(output, context.transpose(1, 2))
        attn = F.softmax(attn, dim=2)

        output = F.tanh(torch.bmm(attn, context))

        return output, attn


class CatDotProdAttention(nn.Module):
    """Dot-Production Attention concatenated with query values
        Attribute:
            linear (nn.Linear): linear layer to compress output
    """
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim*2, dim)

    def forward(self, output, context):
        """Variation of Dot-Production Method
            1. compute e = q * k
            2. compute prod = softmax(e) * k
            3. concatenate prod with q as output
            4. compute and return tanh(linear_layer(output))

            Args:
                output (batch, 1, hidden): output from decoder rnn
                context (batch, seq, hidden): output from encoder rnn
            Returns:
                output (batch, 1, hidden): modified output
                attn (batch, 1, seq): attention state in this step
        """
        attn = torch.bmm(output, context.transpose(1, 2))
        attn = F.softmax(attn, dim=2)

        prod = torch.bmm(attn, context)
        output = torch.cat([prod, output], dim=2)
        output = F.tanh(self.linear(output))

        return output, attn


def set_attn(attn_type, **kwargs):
    """Return defined attention mechanism
        Args:
            attn_type (str): attention method
        Returns:
            attention object
        Raise ValueError if attn_type is invalid or
        required kwargs don't exist
    """
    valids = {
        "dot-prod": 0,
        "cat-dot-prod": 1
    }

    attn_type = valids.get(attn_type)

    if attn_type is None:
        raise ValueError("Attention type should in {}".format(valids.keys()))

    if attn_type == 0:
        return DotProdAttention()

    elif attn_type == 1:
        dim = kwargs.get("dim")
        if dim is None:
            raise ValueError("Need kwarg 'dim'")
        return CatDotProdAttention(dim)
