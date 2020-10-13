import torch

from model import seq2seq, encoder, decoder


voc_size = 10
hid_size = 50
sos = 0
eos = 1
max_len = 6
bidirectional = True
batches, batch_size = 5, 10

# read data
data = [torch.randint(2, voc_size, size=(batch_size, max_len)) for _ in range(batches)]
targets = [torch.randint(2, voc_size, size=(batch_size, max_len)) for _ in range(batches)]

# set model
encoder = encoder.Encoder(voc_size=voc_size, hid_size=hid_size, bidirectional=bidirectional)
decoder = decoder.Decoder(voc_size=voc_size, hid_size=hid_size, attn_type="dot-prod",
                          max_len=max_len, sos=sos, eos=eos, bidirectional=bidirectional)
my_model = seq2seq.Seq2seq(encoder, decoder)

# set loss function and optimizer
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(my_model.parameters(), lr=0.001)


# Train process
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

