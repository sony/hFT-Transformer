#! python

import torch
import torch.nn as nn

##
## Model
##
class Model_SPEC2MIDI(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder_spec2midi = encoder
        self.decoder_spec2midi = decoder

    def forward(self, input_spec):
        #input_spec = [batch_size, n_bin, margin+n_frame+margin] (8, 256, 192)
        #print('Model_SPEC2MIDI(0) input_spec: '+str(input_spec.shape))

        enc_vector = self.encoder_spec2midi(input_spec)
        #enc_freq = [batch_size, n_frame, n_bin, hid_dim] (8, 128, 256, 256)
        #print('Model_SPEC2MIDI(1) enc_vector: '+str(enc_vector.shape))

        output_onset_A, output_offset_A, output_mpe_A, output_velocity_A, attention, output_onset_B, output_offset_B, output_mpe_B, output_velocity_B = self.decoder_spec2midi(enc_vector)
        #output_onset_A = [batch_size, n_frame, n_note] (8, 128, 88)
        #output_onset_B = [batch_size, n_frame, n_note] (8, 128, 88)
        #output_velocity_A = [batch_size, n_frame, n_note, n_velocity] (8, 128, 88, 128)
        #output_velocity_B = [batch_size, n_frame, n_note, n_velocity] (8, 128, 88, 128)
        #attention = [batch_size, n_frame, n_heads, n_note, n_bin] (8, 128, 4, 88, 256)
        #print('Model_SPEC2MIDI(2) output_onset_A: '+str(output_onset_A.shape))
        #print('Model_SPEC2MIDI(2) output_onset_B: '+str(output_onset_B.shape))
        #print('Model_SPEC2MIDI(2) output_velocity_A: '+str(output_velocity_A.shape))
        #print('Model_SPEC2MIDI(2) output_velocity_B: '+str(output_velocity_B.shape))
        #print('Model_SPEC2MIDI(2) attention: '+str(attention.shape))

        return output_onset_A, output_offset_A, output_mpe_A, output_velocity_A, attention, output_onset_B, output_offset_B, output_mpe_B, output_velocity_B


##
## Encoder
##
class Encoder_SPEC2MIDI(nn.Module):
    def __init__(self, n_margin, n_frame, n_bin, cnn_channel, cnn_kernel, hid_dim, n_layers, n_heads, pf_dim, dropout, device):
        super().__init__()

        self.device = device
        self.n_frame = n_frame
        self.n_bin = n_bin
        self.cnn_channel = cnn_channel
        self.cnn_kernel = cnn_kernel
        self.hid_dim = hid_dim
        self.conv = nn.Conv2d(1, self.cnn_channel, kernel_size=(1, self.cnn_kernel))
        self.n_proc = n_margin * 2 + 1
        self.cnn_dim = self.cnn_channel * (self.n_proc - (self.cnn_kernel - 1))
        self.tok_embedding_freq = nn.Linear(self.cnn_dim, hid_dim)
        self.pos_embedding_freq = nn.Embedding(n_bin, hid_dim)
        self.layers_freq = nn.ModuleList([EncoderLayer(hid_dim, n_heads, pf_dim, dropout, device) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.scale_freq = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, spec_in):
        #spec_in = [batch_size, n_bin, n_margin+n_frame+n_margin] (8, 256, 192) (batch_size=8, n_bins=256, margin=32/n_frame=128)
        #print('Encoder_SPEC2MIDI(0) spec_in: '+str(spec_in.shape))
        batch_size = spec_in.shape[0]

        spec = spec_in.unfold(2, self.n_proc, 1).permute(0, 2, 1, 3).contiguous()
        #spec = [batch_size, n_frame, n_bin, n_proc] (8, 128, 256, 65) (batch_size=8, n_frame=128, n_bins=256, n_proc=65)
        #print('Encoder_SPEC2MIDI(1) spec: '+str(spec.shape))

        # CNN 1D
        spec_cnn = spec.reshape(batch_size*self.n_frame, self.n_bin, self.n_proc).unsqueeze(1)
        #spec = [batch_size*n_frame, 1, n_bin, n_proc] (8*128, 1, 256, 65) (batch_size=128, 1, n_frame, n_bins=256, n_proc=65)
        #print('Encoder_SPEC2MIDI(2) spec_cnn: '+str(spec_cnn.shape))
        spec_cnn = self.conv(spec_cnn).permute(0, 2, 1, 3).contiguous()
        # spec_cnn: [batch_size*n_frame, n_bin, cnn_channel, n_proc-(cnn_kernel-1)] (8*128, 256, 4, 61)
        #print('Encoder_SPEC2MIDI(2) spec_cnn: '+str(spec_cnn.shape))

        ##
        ## frequency
        ##
        spec_cnn_freq = spec_cnn.reshape(batch_size*self.n_frame, self.n_bin, self.cnn_dim)
        # spec_cnn_freq: [batch_size*n_frame, n_bin, cnn_channel, (n_proc)-(cnn_kernel-1)] (8*128, 256, 244)
        #print('Encoder_SPEC2MIDI(3) spec_cnn_freq: '+str(spec_cnn_freq.shape))

        # embedding
        spec_emb_freq = self.tok_embedding_freq(spec_cnn_freq)
        # spec_emb_freq: [batch_size*n_frame, n_bin, hid_dim] (8*128, 256, 256)
        #print('Encoder_SPEC2MIDI(4) spec_emb_freq: '+str(spec_emb_freq.shape))

        # position coding
        pos_freq = torch.arange(0, self.n_bin).unsqueeze(0).repeat(batch_size*self.n_frame, 1).to(self.device)
        #pos_freq = [batch_size, n_frame, n_bin] (8*128, 256)
        #print('Encoder_SPEC2MIDI(5) pos_freq: '+str(pos_freq.shape))

        # embedding
        spec_freq = self.dropout((spec_emb_freq * self.scale_freq) + self.pos_embedding_freq(pos_freq))
        #spec_freq = [batch_size*n_frame, n_bin, hid_dim] (8*128, 256, 256)
        #print('Encoder_SPEC2MIDI(6) spec_freq: '+str(spec_freq.shape))

        # transformer encoder
        for layer_freq in self.layers_freq:
            spec_freq = layer_freq(spec_freq)
        spec_freq = spec_freq.reshape(batch_size, self.n_frame, self.n_bin, self.hid_dim)
        #spec_freq = [batch_size, n_frame, n_bin, hid_dim] (8, 128, 256, 256)
        #print('Encoder_SPEC2MIDI(7) spec_freq: '+str(spec_freq.shape))

        return spec_freq


##
## Decoder
##
class Decoder_SPEC2MIDI(nn.Module):
    def __init__(self, n_frame, n_bin, n_note, n_velocity, hid_dim, n_layers, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.device = device
        self.n_note = n_note
        self.n_frame = n_frame
        self.n_velocity = n_velocity
        self.n_bin = n_bin
        self.hid_dim = hid_dim
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

        # CAfreq
        self.pos_embedding_freq = nn.Embedding(n_note, hid_dim)
        self.layer_zero_freq = DecoderLayer_Zero(hid_dim, n_heads, pf_dim, dropout, device)
        self.layers_freq = nn.ModuleList([DecoderLayer(hid_dim, n_heads, pf_dim, dropout, device) for _ in range(n_layers-1)])

        self.fc_onset_freq = nn.Linear(hid_dim, 1)
        self.fc_offset_freq = nn.Linear(hid_dim, 1)
        self.fc_mpe_freq = nn.Linear(hid_dim, 1)
        self.fc_velocity_freq = nn.Linear(hid_dim, self.n_velocity)

        # SAtime
        self.scale_time = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        self.pos_embedding_time = nn.Embedding(n_frame, hid_dim)
        #self.layers_time = nn.ModuleList([DecoderLayer(hid_dim, n_heads, pf_dim, dropout, device) for _ in range(n_layers)])
        self.layers_time = nn.ModuleList([EncoderLayer(hid_dim, n_heads, pf_dim, dropout, device) for _ in range(n_layers)])

        self.fc_onset_time = nn.Linear(hid_dim, 1)
        self.fc_offset_time = nn.Linear(hid_dim, 1)
        self.fc_mpe_time = nn.Linear(hid_dim, 1)
        self.fc_velocity_time = nn.Linear(hid_dim, self.n_velocity)

    def forward(self, enc_spec):
        batch_size = enc_spec.shape[0]
        enc_spec = enc_spec.reshape([batch_size*self.n_frame, self.n_bin, self.hid_dim])
        #enc_spec = [batch_size*n_frame, n_bin, hid_dim] (8*128, 256, 256)
        #print('Decoder_SPEC2MIDI(0) enc_spec: '+str(enc_spec.shape))

        ##
        ## CAfreq freq(256)/note(88)
        ##
        pos_freq = torch.arange(0, self.n_note).unsqueeze(0).repeat(batch_size*self.n_frame, 1).to(self.device)
        midi_freq = self.pos_embedding_freq(pos_freq)
        #pos_freq = [batch_size*n_frame, n_note] (8*128, 88)
        #midi_freq = [batch_size, n_note, hid_dim] (8*128, 88, 256)
        #print('Decoder_SPEC2MIDI(1) pos_freq: '+str(pos_freq.shape))
        #print('Decoder_SPEC2MIDI(1) midi_freq: '+str(midi_freq.shape))

        midi_freq, attention_freq = self.layer_zero_freq(enc_spec, midi_freq)
        for layer_freq in self.layers_freq:
            midi_freq, attention_freq = layer_freq(enc_spec, midi_freq)
        dim = attention_freq.shape
        attention_freq = attention_freq.reshape([batch_size, self.n_frame, dim[1], dim[2], dim[3]])
        #midi_freq = [batch_size*n_frame, n_note, hid_dim] (8*128, 88, 256)
        #attention_freq = [batch_size, n_frame, n_heads, n_note, n_bin] (8, 128, 4, 88, 256)
        #print('Decoder_SPEC2MIDI(2) midi_freq: '+str(midi_freq.shape))
        #print('Decoder_SPEC2MIDI(2) attention_freq: '+str(attention_freq.shape))

        ## output(freq)
        output_onset_freq = self.sigmoid(self.fc_onset_freq(midi_freq).reshape([batch_size, self.n_frame, self.n_note]))
        output_offset_freq = self.sigmoid(self.fc_offset_freq(midi_freq).reshape([batch_size, self.n_frame, self.n_note]))
        output_mpe_freq = self.sigmoid(self.fc_mpe_freq(midi_freq).reshape([batch_size, self.n_frame, self.n_note]))
        output_velocity_freq = self.fc_velocity_freq(midi_freq).reshape([batch_size, self.n_frame, self.n_note, self.n_velocity])
        #output_onset_freq = [batch_size, n_frame, n_note] (8, 128, 88)
        #output_offset_freq = [batch_size, n_frame, n_note] (8, 128, 88)
        #output_mpe_freq = [batch_size, n_frame, n_note] (8, 128, 88)
        #output_velocity_freq = [batch_size, n_frame, n_note, n_velocity] (8, 128, 88, 128)
        #print('Decoder_SPEC2MIDI(3) output_onset_freq: '+str(output_onset_freq.shape))
        #print('Decoder_SPEC2MIDI(3) output_offset_freq: '+str(output_offset_freq.shape))
        #print('Decoder_SPEC2MIDI(3) output_mpe_freq: '+str(output_mpe_freq.shape))
        #print('Decoder_SPEC2MIDI(3) output_velocity_freq: '+str(output_velocity_freq.shape))

        ##
        ## SAtime time(64)
        ##
        #midi_time: [batch_size*n_frame, n_note, hid_dim] -> [batch_size*n_note, n_frame, hid_dim]
        midi_time = midi_freq.reshape([batch_size, self.n_frame, self.n_note, self.hid_dim]).permute(0, 2, 1, 3).contiguous().reshape([batch_size*self.n_note, self.n_frame, self.hid_dim])
        pos_time = torch.arange(0, self.n_frame).unsqueeze(0).repeat(batch_size*self.n_note, 1).to(self.device)
        midi_time = self.dropout((midi_time * self.scale_time) + self.pos_embedding_time(pos_time))
        #pos_time = [batch_size*n_note, n_frame] (8*88, 128)
        #midi_time = [batch_size*n_note, n_frame, hid_dim] (8*88, 128, 256)
        #print('Decoder_SPEC2MIDI(4) pos_time: '+str(pos_time.shape))
        #print('Decoder_SPEC2MIDI(4) midi_time: '+str(midi_time.shape))

        for layer_time in self.layers_time:
            midi_time = layer_time(midi_time)
        #midi_time = [batch_size*n_note, n_frame, hid_dim] (8*88, 128, 256)
        #print('Decoder_SPEC2MIDI(5) midi_time: '+str(midi_time.shape))

        ## output(time)
        output_onset_time = self.sigmoid(self.fc_onset_time(midi_time).reshape([batch_size, self.n_note, self.n_frame]).permute(0, 2, 1).contiguous())
        output_offset_time = self.sigmoid(self.fc_offset_time(midi_time).reshape([batch_size, self.n_note, self.n_frame]).permute(0, 2, 1).contiguous())
        output_mpe_time = self.sigmoid(self.fc_mpe_time(midi_time).reshape([batch_size, self.n_note, self.n_frame]).permute(0, 2, 1).contiguous())
        output_velocity_time = self.fc_velocity_time(midi_time).reshape([batch_size, self.n_note, self.n_frame, self.n_velocity]).permute(0, 2, 1, 3).contiguous()
        #output_onset_time = [batch_size, n_frame, n_note] (8, 128, 88)
        #output_offset_time = [batch_size, n_frame, n_note] (8, 128, 88)
        #output_mpe_time = [batch_size, n_frame, n_note] (8, 128, 88)
        #output_velocity_time = [batch_size, n_frame, n_note, n_velocity] (8, 128, 88, 128)
        #print('Decoder_SPEC2MIDI(6) output_onset_time: '+str(output_onset_time.shape))
        #print('Decoder_SPEC2MIDI(6) output_offset_time: '+str(output_offset_time.shape))
        #print('Decoder_SPEC2MIDI(6) output_mpe_time: '+str(output_mpe_time.shape))
        #print('Decoder_SPEC2MIDI(6) output_velocity_time: '+str(output_velocity_time.shape))

        return output_onset_freq, output_offset_freq, output_mpe_freq, output_velocity_freq, attention_freq, output_onset_time, output_offset_time, output_mpe_time, output_velocity_time


##
## sub functions
##
class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        #src = [batch_size, src_len, hid_dim]

        #self attention
        _src, _ = self.self_attention(src, src, src)
        #dropout, residual connection and layer norm
        src = self.layer_norm(src + self.dropout(_src))
        #src = [batch_size, src_len, hid_dim]

        #positionwise feedforward
        _src = self.positionwise_feedforward(src)
        #dropout, residual and layer norm
        src = self.layer_norm(src + self.dropout(_src))
        #src = [batch_size, src_len, hid_dim]

        return src

class DecoderLayer_Zero(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hid_dim)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_src, trg):
        #trg = [batch_size, trg_len, hid_dim]
        #enc_src = [batch_size, src_len, hid_dim]

        #encoder attention
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src)
        #dropout, residual connection and layer norm
        trg = self.layer_norm(trg + self.dropout(_trg))
        #trg = [batch_size, trg_len, hid_dim]

        #positionwise feedforward
        _trg = self.positionwise_feedforward(trg)
        #dropout, residual and layer norm
        trg = self.layer_norm(trg + self.dropout(_trg))
        #trg = [batch_size, trg_len, hid_dim]
        #attention = [batch_size, n_heads, trg_len, src_len]

        return trg, attention

class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_src, trg):
        #trg = [batch_size, trg_len, hid_dim]
        #enc_src = [batch_size, src_len, hid_dim]

        #self attention
        _trg, _ = self.self_attention(trg, trg, trg)
        #dropout, residual connection and layer norm
        trg = self.layer_norm(trg + self.dropout(_trg))
        #trg = [batch_size, trg_len, hid_dim]

        #encoder attention
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src)
        #dropout, residual connection and layer norm
        trg = self.layer_norm(trg + self.dropout(_trg))
        #trg = [batch_size, trg_len, hid_dim]

        #positionwise feedforward
        _trg = self.positionwise_feedforward(trg)
        #dropout, residual and layer norm
        trg = self.layer_norm(trg + self.dropout(_trg))
        #trg = [batch_size, trg_len, hid_dim]
        #attention = [batch_size, n_heads, trg_len, src_len]

        return trg, attention

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        assert hid_dim % n_heads == 0
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value):
        batch_size = query.shape[0]
        #query = [batch_size, query_len, hid_dim]
        #key = [batch_size, key_len, hid_dim]
        #value = [batch_size, value_len, hid_dim]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        #Q = [batch_size, query_len, hid_dim]
        #K = [batch_size, key_len, hid_dim]
        #V = [batch_size, value_len, hid_dim]

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        #Q = [batch_size, n_heads, query_len, head_dim]
        #K = [batch_size, n_heads, key_len, head_dim]
        #V = [batch_size, n_heads, value_len, head_dim]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        #energy = [batch_size, n_heads, seq len, seq len]

        attention = torch.softmax(energy, dim = -1)
        #attention = [batch_size, n_heads, query_len, key_len]

        x = torch.matmul(self.dropout(attention), V)
        #x = [batch_size, n_heads, seq len, head_dim]

        x = x.permute(0, 2, 1, 3).contiguous()
        #x = [batch_size, seq_len, n_heads, head_dim]

        x = x.view(batch_size, -1, self.hid_dim)
        #x = [batch_size, seq_len, hid_dim]

        x = self.fc_o(x)
        #x = [batch_size, seq_len, hid_dim]

        return x, attention

class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #x = [batch_size, seq_len, hid_dim]

        x = self.dropout(torch.relu(self.fc_1(x)))
        #x = [batch_size, seq_len, pf dim]

        x = self.fc_2(x)
        #x = [batch_size, seq_len, hid_dim]

        return x
