import random
import torch
from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        output, (hidden, cell) = self.rnn(embedded)
        return output, hidden, cell


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        # <YOUR CODE HERE>
        self.attn = nn.Linear(in_features = enc_hid_dim + dec_hid_dim, out_features = dec_hid_dim)
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(in_features = dec_hid_dim, out_features = 1)
        self.softmax = nn.Softmax(dim = 0)
        
    def forward(self, hidden, encoder_outputs):
        # <YOUR CODE HERE>
        enc_len = encoder_outputs.shape[0]
        hidden = torch.tensor(hidden)

        #repeat previous decoder hidden state
        hiddens = hidden.expand(enc_len * hidden.shape[0], -1, -1)

        #concatenate
        concatenated_for_attn = torch.cat(
            [
            hiddens,
            encoder_outputs
            ],
            dim = 2)

        #find attn
        attn_1 = self.attn(concatenated_for_attn)

        #count energy
        energy = self.tanh(attn_1)

        #find a*_t = v * E_t as a result of dense layer
        attn_weights = self.fc(energy)

        #do softmax
        attn_res = self.softmax(attn_weights)

        #to dimension
        attn_res = attn_res.permute((1, 2, 0))
        
        return attn_res
    
    
class DecoderWithAttention(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, emb_dim)

        # <YOUR CODE HERE>
        self.rnn = nn.GRU(input_size = emb_dim + dec_hid_dim, hidden_size  = dec_hid_dim, num_layers = 1, dropout = dropout)
        
        self.out = nn.Linear(in_features = dec_hid_dim * 2 + emb_dim, out_features = output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs):
        # <YOUR CODE HERE>
        input = input.unsqueeze(0)
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)

        # find attention vector
        attention_vector = self.attention(hidden, encoder_outputs)

        # transpose to get correct dimensions
        encoder_outputs = encoder_outputs.permute((1, 0, 2))
        
        #weighted source vector
        weights = torch.bmm(attention_vector, encoder_outputs)

        # to dimendion
        weights = weights.permute((1, 0, 2))

        #concatenation for GRU
        concatenated_for_gru = torch.cat(
            [
            embedded,
            weights
            ],
            dim = 2)

        output, hidden = self.rnn(concatenated_for_gru, hidden)

        #concatenation for the last linear layer
        concatenated_for_out = torch.cat(
            [
            embedded,
            weights,
            output
            ],
            dim = 2)

        prediction = self.out(concatenated_for_out.squeeze(0))

        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.dec_hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        #src = [src sent len, batch size]
        #trg = [trg sent len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        # Again, now batch is the first dimention instead of zero
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        enc_states, hidden, cell = self.encoder(src)
        
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]
        
        for t in range(1, max_len):

            output, hidden = self.decoder(input, hidden, enc_states)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            input = (trg[t] if teacher_force else top1)
        
        return outputs
