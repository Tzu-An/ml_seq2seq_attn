"""Encoder Object"""
import torch.nn as nn

from .rnn_base import RNN

class Encoder(RNN):
    """Encoder of sequence-to-sequence model
        Attributes:
            hid_size (int): size of hidden vectors
            embedding (nn.Embedding): Embedding layer which maps the
                                      index of tokens to tensors
            rnn (nn.LSTM or nn.GRU): RNN layer
    """
    def __init__(self, voc_size, hid_size, layers=1, rnn_type="lstm",
                 bidirectional=False):
        """Initialize an encoder object
            Args:
                voc_size (int): vocabulary size
                hid_size (int): size of hidden vector
                layers (int): number of layers in rnn cells
                rnn_type (str): Type of RNN
                bidirectional (bool): bidirectional encoder or not
        """
        super().__init__(rnn_type, hid_size, hid_size, layers,
                         bidirectional=bidirectional)

        self.embedding = nn.Embedding(voc_size, hid_size)

    def forward(self, inp):
        """Encoding process
            1. Embed the input
            2. Feed the embedded input to RNN

            Args:
                inp (torch.LongTensor): Indices of tokens
            Returns:
                output (torch.Tensor, size = (batch, 1, hidden)):
                    output of rnn layer
                hidden (tuple of torch.Tensor):
                    hidden states of rnn layer
        """
        embed = self.embedding(inp)
        output, hidden = self.rnn(embed)
        return output, hidden