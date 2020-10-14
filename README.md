# ml_seq2seq_attn
Seq2seq model with attention mechanism (Pytorch)

## This is a basic pytorch implementation of sequence-to-sequence model with attention mechanism

## To add
  - tests for model
  - dropouts
  - more type of attention mechanism
  - decoder for prediction
  - Topk decoder for prediction

## When to use
  - Translation problem
  - Any problem with a sequence of index as input and output

## How to use

### Please add this repo as your submodule, and you can refer to example.py to design your own training process

```python
import torch
from ml_seq2seq_attn.model import seq2seq, encoder, decoder

# Constants you might need
voc_size = 10
hid_size = 50
sos = 0
eos = 1
max_len = 6
bidirectional = True
batches, batch_size = 5, 10

# read data (generate mock data)
data = [torch.randint(2, voc_size, size=(batch_size, max_len)) for _ in range(batches)]
targets = [torch.randint(2, voc_size, size=(batch_size, max_len)) for _ in range(batches)]

# set your model
encoder = encoder.Encoder(voc_size=voc_size, hid_size=hid_size, bidirectional=bidirectional)
decoder = decoder.Decoder(voc_size=voc_size, hid_size=hid_size, attn_type="dot-prod",
                          max_len=max_len, sos=sos, eos=eos, bidirectional=bidirectional)
my_model = seq2seq.Seq2seq(encoder, decoder)

# set loss function and optimizer
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(my_model.parameters(), lr=0.001)

# Training process, updated parameters for each batch

for epoch in range(5):

    for batch, batch_data in enumerate(data):
        optimizer.zero_grad()
        output = my_model.forward(batch_data)
        probs = torch.cat(output["LogProbs"], dim=1)

        for idx, tar_seq in enumerate(targets[batch]):
            loss = loss_function(probs[idx], tar_seq)
            if idx == 0:
                batch_loss = loss
            else:
                batch_loss += loss

        batch_loss.backward()
        optimizer.step()

    print("epoch {}, loss on last batch: {}".format(epoch, batch_loss))

```
