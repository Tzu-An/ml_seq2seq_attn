"""Decodr Object"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rnn_base import RNN
from .attention import *


class Decoder(RNN):
    """Decoder of sequence-to-sequence model
        Attributes:
            hid_size (int): hidden size
            max_len (int): maximum length of sequences
            sos (int): index of start of sequences
            eos (int): index of end of sequences
            bidirectional (bool): bidirectional encoder or not
            rnn: RNN cell
            attention: attention mechanism
            embed: embedding layer for decoder inputs
            linear: a linear layer maps hidden vector to output
    """
    def __init__(self, voc_size, hid_size, max_len, sos, eos,
                 rnn_type="lstm", layers=1, bidirectional=False):
        """Initialize decoder instance
            Args:
                voc_size (int): vocabulary size
                hid_size (int): hidden size
                max_len (int): maximum length of sequences
                sos (int): index of start of sequences
                eos (int): index of end of sequences
                layers (int): layers of rnn cells
                bidirectional (bool): bidirectional encoder or not
        """
        self._validate_additional_args(
            voc_size, max_len, sos,eos, bidirectional)

        super().__init__(rnn_type, hid_size,
                         hid_size * (bidirectional+1),
                         layers)
        self.max_len = max_len
        self.sos = sos
        self.eos = eos
        self.bidirectional = bidirectional

        self.attention = DotProdAttention()
        self.embed = nn.Embedding(voc_size, hid_size)
        self.linear = nn.Linear(self.hid_size, voc_size)

    @staticmethod
    def _validate_additional_args(voc_size, max_len, sos, eos, bidirectional):
        """Validate additional arguments
            Args:
                refers to __init__
        """
        if not isinstance(voc_size, int) or voc_size <= 0:
            raise ValueError("Input 'voc_size' should be a positive integer")
        if not isinstance(max_len, int) or max_len <= 0:
            raise ValueError("Input 'max_len' should be a positive integer")
        if not isinstance(sos, int) or sos < 0:
            raise ValueError("Input 'sos' should be a non negative integer")
        if not isinstance(eos, int) or eos < 0:
            raise ValueError("Input 'eos' should be a non negative integer")
        if sos == eos:
            raise ValueError("Input 'eos' and 'sos' should be different")
        if not isinstance(bidirectional, bool):
            raise ValueError("Input 'bidirectional' should be boolean value")

    def _init_input(self, batch_size):
        """Initialize input of decoder
            Args:
                batch_size (int): batch size
            Returns ret (torch.LongTensor):
                start of sequences
        """
        ret = torch.LongTensor([[self.sos]] * batch_size)
        if torch.cuda.is_available():
            return ret.cuda()
        return ret

    def _init_hidden(self, enc_hidden):
        """Initialize hidden states of decoder
            Args:
                enc_hidden: hidden states of encoder
            Returns:
                dec_hidden: hidden states of decoder
        """
        if enc_hidden is None:
            return None
        if self.bidirectional:
            return tuple([self._cat(state) for state in enc_hidden])
        return enc_hidden

    @staticmethod
    def _cat(state):
        """Concatenate 2 directional hidden(cell) states
            origin: num_layers * num_direcitons, batch, hidden_size
            after: num_layers, batch, num_directions * hidden_size

            Args:
                state: hidden (cell) states
            Returns:
                concatenated hidden (cell) states
        """
        end = state.size(0)
        return torch.cat([state[0:end:2], state[1:end:2]], dim=2)

    def move_a_step(self, step_input, step_hidden, enc_output):
        """Move a step forward
            1. embed the input(output from previous step)
            2. put embedded input and hidden states
               (from previous step) into rnn cell
            3. compute attension with encoder output
            4. a linear layer maps hidden vectors to vocabulary size
            5. compute log softmax on the output

            Args:
                step_input (torch.LongTensor, size=(batch, 1)):
                    Decoded output from previous rnn step
                step_hidden (tuple of torch.Tensor):
                    hidden states from previous rnn step
                enc_output (torch.Tensor, size=(batch, seq, hidden)):
                    outputs of encoder
            Returns:
                output (torch.Tensor, size = (batch, 1, hidden)):
                    output of this rnn step
                hidden (tuple of torch.Tensor):
                    hidden state of this rnn step
                attn: attention of this rnn step
        """

        embedded = self.embed(step_input)
        output, hidden = self.rnn(embedded, step_hidden)
        output, attn = self.attention(output, enc_output)
        output = self.linear(output)
        output = F.log_softmax(output, dim=2)

        return output, hidden, attn

    def forward(self, enc_output, enc_hidden):
        """Decoding process
            1. initialize decoder input with start of sequence index
            2. initialize hidden states with encoder hidden states
            3. iteratively compute outcomes

            Args:
                enc_output (torch.Tensor, size=(batch, seq, hidden)):
                    outputs of encoder
                enc_hidden (tuple of torch.Tensor):
                    hidden states of encoder, None is accepted.
            Returns ret_dict (dict):
                - Symbols (list of torch.LongTensor): index of tokens
                - LogProbs (list of torch.Tensor): log softmax values
        """

        ret_dict = {"Symbols": list(), "LogProbs": list()}

        batch_size = enc_output.size(0)
        step_input = self._init_input(batch_size)
        step_hidden = self._init_hidden(enc_hidden)

        for _ in range(self.max_len):
            step_output, step_hidden, step_attn = self.move_a_step(
                step_input, step_hidden, enc_output)

            symbols = step_output.topk(1, dim=2)[1].squeeze(2)
            ret_dict["LogProbs"].append(step_output)
            ret_dict["Symbols"].append(symbols)
            step_input = symbols

        return ret_dict



