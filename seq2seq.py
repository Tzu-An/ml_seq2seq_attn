"""Main structure of seq2seq model"""
import torch.nn as nn


class Seq2seq(nn.Module):
    """Main structure of a sequence-to-sequence model
        Attributes:
            encoder (RNN): RNN encoder object
            decoder (RNN): RNN decoder object
    """
    def __init__(self, encoder, decoder):
        """Initialize sequence-to-sequence model
            Args:
                encoder (RNN): RNN encoder object
                decoder (RNN): RNN decoder object
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, inp):
        """Forward process of seq2seq model
            Args:
                inp (torch.LongTensor): Index of tokens
            Returns:
                dec_output: outputs from decoder
        """
        enc_output, enc_hidden = self.encoder(inp)
        dec_output = self.decoder(enc_output, enc_hidden)

        return dec_output
