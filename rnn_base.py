"""RNN base class"""
import torch.nn as nn


VALID_RNN = {
    "lstm": nn.LSTM,
    "gru": nn.GRU
}


class RNN(nn.Module):
    """RNN base class
        Attributes:
            rnn: RNN layer
            hid_size (int): size of hidden vectors
    """
    def __init__(self, *args, **kwargs):
        """Initialize an RNN object
            Args:
                rnn_type (str): type of RNN
                inp_size (int): input size of RNN
                hid_size (int): hidden size of RNN
                layers (int): layers in RNN
            Kwargs:
                bidirectional (bool): bidirectional RNN or not
        """
        super().__init__()
        rnn_type, inp_size, hid_size, layers = args
        bidir = kwargs.get("bidirectional", False)
        self._validate_args(rnn_type, inp_size, hid_size, layers, bidir)
        self.rnn = self._set_rnn(rnn_type, inp_size, hid_size, layers, bidir)
        self.hid_size = hid_size

    @staticmethod
    def _validate_args(rnn_type, inp_size, hid_size, layers, bidir):
        """Validate Arguments for initializing RNN
            Args:
                refers to __init__
        """
        if not isinstance(rnn_type, str) or rnn_type.lower() not in VALID_RNN:
            raise ValueError("Input 'rnn_type' should be a string")
        if not isinstance(inp_size, int) or inp_size <= 0:
            raise ValueError("Input 'inp_size' should be a positive integer")
        if not isinstance(hid_size, int) or hid_size <= 0:
            raise ValueError("Input 'hid_size' should be a positive integer")
        if not isinstance(layers, int) or layers < 1:
            raise ValueError("Input 'layers' should be a positive integer")
        if not isinstance(bidir, bool):
            raise ValueError("Input 'bidirectional' should be boolean value")

    @staticmethod
    def _set_rnn(rnn_type, inp_size, hid_size, layers, bidir):
        """Set RNN cells
            Args:
                refers to __init__
            Returns RNN object
        """
        return VALID_RNN[rnn_type.lower()](inp_size, hid_size,
                                           num_layers=layers,
                                           bidirectional=bidir,
                                           batch_first=True)

    def forward(self, *args, **kwargs):
        """"""
        raise NotImplementedError()
