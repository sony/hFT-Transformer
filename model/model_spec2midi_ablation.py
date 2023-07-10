#! python

import torch
import torch.nn as nn

##
## Model (single output)
##
# 1FDN: Encoder_CNNtime_SAfreq / Decoder_CAfreq
class Model_single(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder_spec2midi = encoder
        self.decoder_spec2midi = decoder

    def forward(self, input_spec):
        #input_spec = [batch_size, n_bin, margin+n_frame+margin] (8, 256, 192)
        #print('Model_single(0) input_spec: '+str(input_spec.shape))

        enc_vector = self.encoder_spec2midi(input_spec)
        #enc_freq = [batch_size, n_frame, n_bin, hid_dim] (8, 128, 256, 256)
        #print('Model_single(1) enc_vector: '+str(enc_vector.shape))

        output_onset, output_offset, output_mpe, output_velocity = self.decoder_spec2midi(enc_vector)
        #output_onset = [batch_size, n_frame, n_note] (8, 128, 88)
        #output_velocity = [batch_size, n_frame, n_note, n_velocity] (8, 128, 88, 128)
        #print('Model_single(2) output_onset: '+str(output_onset.shape))
        #print('Model_single(2) output_velocity: '+str(output_velocity.shape))

        return output_onset, output_offset, output_mpe, output_velocity


##
## Model (combination output)
##
# 1FDT: Encoder_CNNtime_SAfreq / Decoder_CAfreq_SAtime
# 1FLT: Encoder_CNNtime_SAfreq / Decoder_linear_SAtime
# 2FDT: Encoder_CNNblock_SAfreq / Decoder_CAfreq_SAtime
class Model_combination(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder_spec2midi = encoder
        self.decoder_spec2midi = decoder

    def forward(self, input_spec):
        #input_spec = [batch_size, n_bin, margin+n_frame+margin] (8, 256, 192)
        #print('Model_combination(0) input_spec: '+str(input_spec.shape))

        enc_vector = self.encoder_spec2midi(input_spec)
        #enc_freq = [batch_size, n_frame, n_bin, hid_dim] (8, 128, 256, 256)
        #print('Model_combination(1) enc_vector: '+str(enc_vector.shape))

        output_onset_A, output_offset_A, output_mpe_A, output_velocity_A, output_onset_B, output_offset_B, output_mpe_B, output_velocity_B = self.decoder_spec2midi(enc_vector)
        #output_onset_A = [batch_size, n_frame, n_note] (8, 128, 88)
        #output_velocity_A = [batch_size, n_frame, n_note, n_velocity] (8, 128, 88, 128)
        #print('Model_combination(2) output_onset_A: '+str(output_onset_A.shape))
        #print('Model_combination(2) output_velocity_A: '+str(output_velocity_A.shape))
        #print('Model_combination(2) output_onset_B: '+str(output_onset_B.shape))
        #print('Model_combination(2) output_velocity_B: '+str(output_velocity_B.shape))

        return output_onset_A, output_offset_A, output_mpe_A, output_velocity_A, output_onset_B, output_offset_B, output_mpe_B, output_velocity_B


##
## Encoder
##
# Encoder_CNNtime_SAfreq
# Encoder_CNNblock_SAfreq
##
## Encoder CNN(time)+SA(freq)
##
class Encoder_CNNtime_SAfreq(nn.Module):
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
        #print('Encoder_CNNtime_SAfreq(0) spec_in: '+str(spec_in.shape))
        batch_size = spec_in.shape[0]

        # CNN
        spec_cnn = self.conv(spec_in.unsqueeze(1))
        #spec_cnn: [batch_size, cnn_channel, n_bin, n_margin+n_frame+n_margin-(cnn_kernel-1)] (8, 4, 256, 188)
        #print('Encoder_CNNtime_SAfreq(1) spec_cnn: '+str(spec_cnn.shape))

        # n_frame block
        spec_cnn = spec_cnn.unfold(3, 61, 1).permute(0, 3, 2, 1, 4).contiguous().reshape([batch_size*self.n_frame, self.n_bin, self.cnn_dim])
        #spec_cnn: [batch_size*n_frame, n_bin, cnn_dim] (8*128, 256, 244)
        #print('Encoder_CNNtime_SAfreq(2) spec_cnn: '+str(spec_cnn.shape))

        # embedding
        spec_emb_freq = self.tok_embedding_freq(spec_cnn)
        # spec_emb_freq: [batch_size*n_frame, n_bin, hid_dim] (8*128, 256, 256)
        #print('Encoder_CNNtime_SAfreq(3) spec_emb_freq: '+str(spec_emb_freq.shape))

        # position coding
        pos_freq = torch.arange(0, self.n_bin).unsqueeze(0).repeat(batch_size*self.n_frame, 1).to(self.device)
        #pos_freq = [batch_size*n_frame, n_bin] (8*128, 256)
        #print('Encoder_CNNtime_SAfreq(4) pos_freq: '+str(pos_freq.shape))

        # embedding
        spec_freq = self.dropout((spec_emb_freq * self.scale_freq) + self.pos_embedding_freq(pos_freq))
        #spec_freq = [batch_size*n_frame, n_bin, hid_dim] (8*128, 256, 256)
        #print('Encoder_CNNtime_SAfreq(5) spec_freq: '+str(spec_freq.shape))

        # transformer encoder
        for layer_freq in self.layers_freq:
            spec_freq = layer_freq(spec_freq)
        spec_freq = spec_freq.reshape([batch_size, self.n_frame, self.n_bin, self.hid_dim])
        #spec_freq = [batch_size, n_frame, n_bin, hid_dim] (8, 128, 256, 256)
        #print('Encoder_CNNtime_SAfreq(6) spec_freq: '+str(spec_freq.shape))

        return spec_freq


##
## Encoder CNN(block)+SA(freq)
##
class Encoder_CNNblock_SAfreq(nn.Module):
    def __init__(self, n_margin, n_frame, n_bin, hid_dim, n_layers, n_heads, pf_dim, dropout, dropout_convblock, device):
        super().__init__()

        self.device = device
        self.n_frame = n_frame
        self.n_bin = n_bin
        self.hid_dim = hid_dim

        k = 3
        p = 1
        # ConvBlock1
        layers_conv_1 = []
        ch1 = 48
        layers_conv_1.append(nn.Conv2d(1, ch1, kernel_size=k, stride=1, padding=p))
        layers_conv_1.append(nn.BatchNorm2d(ch1))
        layers_conv_1.append(nn.ReLU(True))
        layers_conv_1.append(nn.Conv2d(ch1, ch1, kernel_size=k, stride=1, padding=p))
        layers_conv_1.append(nn.BatchNorm2d(ch1))
        layers_conv_1.append(nn.ReLU(True))
        layers_conv_1.append(nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0))
        self.conv_1 = nn.Sequential(*layers_conv_1)
        self.dropout_1 = nn.Dropout(dropout_convblock)
        # ConvBlock2
        layers_conv_2 = []
        ch2 = 64
        layers_conv_2.append(nn.Conv2d(ch1, ch2, kernel_size=k, stride=1, padding=p))
        layers_conv_2.append(nn.BatchNorm2d(ch2))
        layers_conv_2.append(nn.ReLU(True))
        layers_conv_2.append(nn.Conv2d(ch2, ch2, kernel_size=k, stride=1, padding=p))
        layers_conv_2.append(nn.BatchNorm2d(ch2))
        layers_conv_2.append(nn.ReLU(True))
        layers_conv_2.append(nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0))
        self.conv_2 = nn.Sequential(*layers_conv_2)
        self.dropout_2 = nn.Dropout(dropout_convblock)
        # ConvBlock3
        layers_conv_3 = []
        ch3 = 96
        layers_conv_3.append(nn.Conv2d(ch2, ch3, kernel_size=k, stride=1, padding=p))
        layers_conv_3.append(nn.BatchNorm2d(ch3))
        layers_conv_3.append(nn.ReLU(True))
        layers_conv_3.append(nn.Conv2d(ch3, ch3, kernel_size=k, stride=1, padding=p))
        layers_conv_3.append(nn.BatchNorm2d(ch3))
        layers_conv_3.append(nn.ReLU(True))
        layers_conv_3.append(nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0))
        self.conv_3 = nn.Sequential(*layers_conv_3)
        self.dropout_3 = nn.Dropout(dropout_convblock)
        # ConvBlock4
        layers_conv_4 = []
        ch4 = 128
        layers_conv_4.append(nn.Conv2d(ch3, ch4, kernel_size=k, stride=1, padding=p))
        layers_conv_4.append(nn.BatchNorm2d(ch4))
        layers_conv_4.append(nn.ReLU(True))
        layers_conv_4.append(nn.Conv2d(ch4, ch4, kernel_size=k, stride=1, padding=p))
        layers_conv_4.append(nn.BatchNorm2d(ch4))
        layers_conv_4.append(nn.ReLU(True))
        layers_conv_4.append(nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0))
        self.conv_4 = nn.Sequential(*layers_conv_4)
        self.dropout_4 = nn.Dropout(dropout_convblock)                            

        self.n_proc = n_margin * 2 + 1
        self.cnn_dim = int(int(int(int(self.n_bin/2)/2)/2)/2)
        self.cnn_channel_A = 16
        self.cnn_channel_B = 8
        self.cnn_out_dim = self.n_proc * self.cnn_channel_B

        self.tok_embedding_freq = nn.Linear(self.cnn_out_dim, hid_dim)
        self.pos_embedding_freq = nn.Embedding(n_bin, hid_dim)
        self.layers_freq = nn.ModuleList([EncoderLayer(hid_dim, n_heads, pf_dim, dropout, device) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.scale_freq = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, spec_in):
        #spec_in = [batch_size, n_bin, n_margin+n_frame+n_margin] (8, 256, 192) (batch_size=8, n_bins=256, margin=32/n_frame=128)
        #print('Encoder_CNNblock_SAfreq(0) spec_in: '+str(spec_in.shape))
        batch_size = spec_in.shape[0]

        # conv blocks
        spec1 = self.dropout_1(self.conv_1(spec_in.permute(0, 2, 1).contiguous().unsqueeze(1)))
        #spec1 = [batch_size, ch1, n_margin+n_frame+n_margin, int(n_bin/2)] (8, 48, 192, 128)
        #print('Encoder_CNNblock_SAfreq(1) spec1: '+str(spec1.shape))

        spec2 = self.dropout_2(self.conv_2(spec1))
        #spec2 = [batch_size, ch2, n_margin+n_frame+n_margin, int(int(n_bin/2)/2)] (8, 64, 192, 64)
        #print('Encoder_CNNblock_SAfreq(2) spec2: '+str(spec2.shape))

        spec3 = self.dropout_3(self.conv_3(spec2))
        #spec3 = [batch_size, ch3, n_margin+n_frame+n_margin, int(int(int(n_bin/2)/2)/2)] (8, 96, 192, 32)
        #print('Encoder_CNNblock_SAfreq(3) spec3: '+str(spec3.shape))

        spec4 = self.dropout_4(self.conv_4(spec3))
        #spec4 = [batch_size, ch4, n_margin+n_frame+n_margin, int(int(int(int(n_bin/2)/2)/2)/2)] (8, 128, 192, 16)
        #print('Encoder_CNNblock_SAfreq(4) spec4: '+str(spec4.shape))

        # n_frame block
        spec5 = spec4.unfold(2, self.n_proc, 1)
        #spec5: [batch_size, ch4, n_frame, 16bin, n_proc] (8, 128, 128, 16, 65)
        #print('Encoder_CNNblock_SAfreq(5) spec5: '+str(spec5.shape))

        spec6 = spec5.permute(0, 2, 3, 1, 4).contiguous()
        #spec6: [batch_size, n_frame, cnn_dim, ch4, n_proc] (8, 128, 16, 128, 65)
        #print('Encoder_CNNblock_SAfreq(6) spec6: '+str(spec6.shape))

        spec7 = spec6.reshape([batch_size, self.n_frame, self.cnn_dim, self.cnn_channel_A, self.cnn_channel_B, self.n_proc])
        #spec7: [batch_size, n_frame, cnn_dim, cnn_channel_A, cnn_channel_B, n_proc] (8, 128, 16, 16, 8, 65)
        #print('Encoder_CNNblock_SAfreq(7) spec7: '+str(spec7.shape))

        spec8 = spec7.reshape([batch_size, self.n_frame, self.n_bin, self.cnn_out_dim])
        #spec8: [batch_size, n_frame, n_bin, cnn_out_dim] (8, 128, 256, 520)
        #print('Encoder_CNNblock_SAfreq(8) spec8: '+str(spec8.shape))

        spec_emb_freq = self.tok_embedding_freq(spec8).reshape([batch_size*self.n_frame, self.n_bin, self.hid_dim])
        # spec_emb_freq: [batch_size*n_frame, n_bin, hid_dim] (8*128, 256, 256)
        #print('Encoder_CNNblock_SAfreq(9) spec_emb_freq: '+str(spec_emb_freq.shape))

        # position coding
        pos_freq = torch.arange(0, self.n_bin).unsqueeze(0).repeat(batch_size*self.n_frame, 1).to(self.device)
        #pos_freq = [batch_size*n_frame, n_bin] (8*128, 256)
        #print('Encoder_CNNblock_SAfreq(10) pos_freq: '+str(pos_freq.shape))

        # embedding
        spec_freq = self.dropout((spec_emb_freq * self.scale_freq) + self.pos_embedding_freq(pos_freq))
        #spec_freq = [batch_size*n_frame, n_bin, hid_dim] (8*128, 256, 256)
        #print('Encoder_CNNblock_SAfreq(11) spec_freq: '+str(spec_freq.shape))

        # transformer encoder
        for layer_freq in self.layers_freq:
            spec_freq = layer_freq(spec_freq)
        spec_freq = spec_freq.reshape(batch_size, self.n_frame, self.n_bin, self.hid_dim)
        #spec_freq = [batch_size, n_frame, n_bin, hid_dim] (8, 128, 256, 256)
        #print('Encoder_CNNblock_SAfreq(12) spec_freq: '+str(spec_freq.shape))

        return spec_freq


##
## Decoder
##
# Decoder_CAfreq
# Decoder_CAfreq_SAtime
# Decoder_linear_SAtime
##
## Decoder CA(freq)
##
class Decoder_CAfreq(nn.Module):
    def __init__(self, n_frame, n_bin, n_note, n_velocity, hid_dim, n_layers, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.device = device
        self.n_note = n_note
        self.n_frame = n_frame
        self.n_velocity = n_velocity
        self.n_bin = n_bin
        self.hid_dim = hid_dim

        self.pos_embedding_freq = nn.Embedding(n_note, hid_dim)
        self.layer_zero_freq = DecoderLayer_Zero(hid_dim, n_heads, pf_dim, dropout, device)
        self.layers_freq = nn.ModuleList([DecoderLayer(hid_dim, n_heads, pf_dim, dropout, device) for _ in range(n_layers-1)])

        self.fc_onset_freq = nn.Linear(hid_dim, 1)
        self.fc_offset_freq = nn.Linear(hid_dim, 1)
        self.fc_mpe_freq = nn.Linear(hid_dim, 1)
        self.fc_velocity_freq = nn.Linear(hid_dim, self.n_velocity)

        self.sigmoid = nn.Sigmoid()

    def forward(self, enc_spec):
        #enc_spec = [batch_size, n_frame, n_bin, hid_dim] (8, 128, 256, 256)
        batch_size = enc_spec.shape[0]

        enc_spec = enc_spec.reshape([batch_size*self.n_frame, self.n_bin, self.hid_dim])
        #enc_spec = [batch_size*n_frame, n_bin, hid_dim] (8*128, 256, 256)
        #print('Decoder_CAfreq(0) enc_spec: '+str(enc_spec.shape))

        ##
        ## CAfreq bin(256)/note(88)
        ##
        pos_freq = torch.arange(0, self.n_note).unsqueeze(0).repeat(batch_size*self.n_frame, 1).to(self.device)
        midi_freq = self.pos_embedding_freq(pos_freq)
        #pos_freq = [batch_size*n_frame, n_note] (8*128, 88)
        #midi_freq = [batch_size*n_frame, n_note, hid_dim] (8*128, 88, 256)
        #print('Decoder_CAfreq(1) pos_freq: '+str(pos_freq.shape))
        #print('Decoder_CAfreq(1) midi_freq: '+str(midi_freq.shape))

        midi_freq, attention_freq = self.layer_zero_freq(enc_spec, midi_freq)
        for layer_freq in self.layers_freq:
            midi_freq, attention_freq = layer_freq(enc_spec, midi_freq)
        dim = attention_freq.shape
        attention_freq = attention_freq.reshape([batch_size, self.n_frame, dim[1], dim[2], dim[3]])
        #midi_freq = [batch_size*n_frame, n_note, hid_dim] (8*128, 88, 256)
        #attention_freq = [batch_size, n_frame, n_heads, n_note, n_bin] (8, 128, 4, 88, 256)
        #print('Decoder_CAfreq(2) midi_freq: '+str(midi_freq.shape))
        #print('Decoder_CAfreq(2) attention_freq: '+str(attention_freq.shape))

        ## output
        output_onset_freq = self.sigmoid(self.fc_onset_freq(midi_freq).reshape([batch_size, self.n_frame, self.n_note]))
        output_offset_freq = self.sigmoid(self.fc_offset_freq(midi_freq).reshape([batch_size, self.n_frame, self.n_note]))
        output_mpe_freq = self.sigmoid(self.fc_mpe_freq(midi_freq).reshape([batch_size, self.n_frame, self.n_note]))
        output_velocity_freq = self.fc_velocity_freq(midi_freq).reshape([batch_size, self.n_frame, self.n_note, self.n_velocity])
        #output_onset_freq = [batch_size, n_frame, n_note] (8, 128, 88)
        #output_offset_freq = [batch_size, n_frame, n_note] (8, 128, 88)
        #output_mpe_freq = [batch_size, n_frame, n_note] (8, 128, 88)
        #output_velocity_freq = [batch_size, n_frame, n_note, n_velocity] (8, 128, 88, 128)
        #print('Decoder_CAfreq(3) output_onset_freq: '+str(output_onset_freq.shape))
        #print('Decoder_CAfreq(3) output_offset_freq: '+str(output_offset_freq.shape))
        #print('Decoder_CAfreq(3) output_mpe_freq: '+str(output_mpe_freq.shape))
        #print('Decoder_CAfreq(3) output_velocity_freq: '+str(output_velocity_freq.shape))

        return output_onset_freq, output_offset_freq, output_mpe_freq, output_velocity_freq


##
## Decoder CA(freq)/SA(time)
##
class Decoder_CAfreq_SAtime(nn.Module):
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
        #print('Decoder_CAfreq_SAtime(0) enc_spec: '+str(enc_spec.shape))

        ##
        ## CAfreq freq(256)/note(88)
        ##
        pos_freq = torch.arange(0, self.n_note).unsqueeze(0).repeat(batch_size*self.n_frame, 1).to(self.device)
        midi_freq = self.pos_embedding_freq(pos_freq)
        #pos_freq = [batch_size*n_frame, n_note] (8*128, 88)
        #midi_freq = [batch_size, n_note, hid_dim] (8*128, 88, 256)
        #print('Decoder_CAfreq_SAtime(1) pos_freq: '+str(pos_freq.shape))
        #print('Decoder_CAfreq_SAtime(1) midi_freq: '+str(midi_freq.shape))

        midi_freq, attention_freq = self.layer_zero_freq(enc_spec, midi_freq)
        for layer_freq in self.layers_freq:
            midi_freq, attention_freq = layer_freq(enc_spec, midi_freq)
        dim = attention_freq.shape
        attention_freq = attention_freq.reshape([batch_size, self.n_frame, dim[1], dim[2], dim[3]])
        #midi_freq = [batch_size*n_frame, n_note, hid_dim] (8*128, 88, 256)
        #attention_freq = [batch_size, n_frame, n_heads, n_note, n_bin] (8, 128, 4, 88, 256)
        #print('Decoder_CAfreq_SAtime(2) midi_freq: '+str(midi_freq.shape))
        #print('Decoder_CAfreq_SAtime(2) attention_freq: '+str(attention_freq.shape))

        ## output(freq)
        output_onset_freq = self.sigmoid(self.fc_onset_freq(midi_freq).reshape([batch_size, self.n_frame, self.n_note]))
        output_offset_freq = self.sigmoid(self.fc_offset_freq(midi_freq).reshape([batch_size, self.n_frame, self.n_note]))
        output_mpe_freq = self.sigmoid(self.fc_mpe_freq(midi_freq).reshape([batch_size, self.n_frame, self.n_note]))
        output_velocity_freq = self.fc_velocity_freq(midi_freq).reshape([batch_size, self.n_frame, self.n_note, self.n_velocity])
        #output_onset_freq = [batch_size, n_frame, n_note] (8, 128, 88)
        #output_offset_freq = [batch_size, n_frame, n_note] (8, 128, 88)
        #output_mpe_freq = [batch_size, n_frame, n_note] (8, 128, 88)
        #output_velocity_freq = [batch_size, n_frame, n_note, n_velocity] (8, 128, 88, 128)
        #print('Decoder_CAfreq_SAtime(3) output_onset_freq: '+str(output_onset_freq.shape))
        #print('Decoder_CAfreq_SAtime(3) output_offset_freq: '+str(output_offset_freq.shape))
        #print('Decoder_CAfreq_SAtime(3) output_mpe_freq: '+str(output_mpe_freq.shape))
        #print('Decoder_CAfreq_SAtime(3) output_velocity_freq: '+str(output_velocity_freq.shape))

        ##
        ## SAtime time(64)
        ##
        #midi_time: [batch_size*n_frame, n_note, hid_dim] -> [batch_size*n_note, n_frame, hid_dim]
        midi_time = midi_freq.reshape([batch_size, self.n_frame, self.n_note, self.hid_dim]).permute(0, 2, 1, 3).contiguous().reshape([batch_size*self.n_note, self.n_frame, self.hid_dim])
        pos_time = torch.arange(0, self.n_frame).unsqueeze(0).repeat(batch_size*self.n_note, 1).to(self.device)
        midi_time = self.dropout((midi_time * self.scale_time) + self.pos_embedding_time(pos_time))
        #pos_time = [batch_size*n_note, n_frame] (8*88, 128)
        #midi_time = [batch_size*n_note, n_frame, hid_dim] (8*88, 128, 256)
        #print('Decoder_CAfreq_SAtime(4) pos_time: '+str(pos_time.shape))
        #print('Decoder_CAfreq_SAtime(4) midi_time: '+str(midi_time.shape))

        for layer_time in self.layers_time:
            midi_time = layer_time(midi_time)
        #midi_time = [batch_size*n_note, n_frame, hid_dim] (8*88, 128, 256)
        #print('Decoder_CAfreq_SAtime(5) midi_time: '+str(midi_time.shape))

        ## output(time)
        output_onset_time = self.sigmoid(self.fc_onset_time(midi_time).reshape([batch_size, self.n_note, self.n_frame]).permute(0, 2, 1).contiguous())
        output_offset_time = self.sigmoid(self.fc_offset_time(midi_time).reshape([batch_size, self.n_note, self.n_frame]).permute(0, 2, 1).contiguous())
        output_mpe_time = self.sigmoid(self.fc_mpe_time(midi_time).reshape([batch_size, self.n_note, self.n_frame]).permute(0, 2, 1).contiguous())
        output_velocity_time = self.fc_velocity_time(midi_time).reshape([batch_size, self.n_note, self.n_frame, self.n_velocity]).permute(0, 2, 1, 3).contiguous()
        #output_onset_time = [batch_size, n_frame, n_note] (8, 128, 88)
        #output_offset_time = [batch_size, n_frame, n_note] (8, 128, 88)
        #output_mpe_time = [batch_size, n_frame, n_note] (8, 128, 88)
        #output_velocity_time = [batch_size, n_frame, n_note, n_velocity] (8, 128, 88, 128)
        #print('Decoder_CAfreq_SAtime(6) output_onset_time: '+str(output_onset_time.shape))
        #print('Decoder_CAfreq_SAtime(6) output_offset_time: '+str(output_offset_time.shape))
        #print('Decoder_CAfreq_SAtime(6) output_mpe_time: '+str(output_mpe_time.shape))
        #print('Decoder_CAfreq_SAtime(6) output_velocity_time: '+str(output_velocity_time.shape))

        return output_onset_freq, output_offset_freq, output_mpe_freq, output_velocity_freq, output_onset_time, output_offset_time, output_mpe_time, output_velocity_time


##
## Decoder linear/SA(time)
##
class Decoder_linear_SAtime(nn.Module):
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

        self.fc_convert = nn.Linear(n_bin, n_note)

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
        enc_spec = enc_spec.permute(0, 1, 3, 2).contiguous().reshape([batch_size*self.n_frame, self.hid_dim, self.n_bin])
        #enc_spec = [batch_size*n_frame, hid_dim, n_bin] (8*128, 256, 256)
        #print('Decoder_linear_SAtime(0) enc_spec: '+str(enc_spec.shape))

        ##
        ## linear bin(256)/note(88)
        ##
        midi_freq = self.fc_convert(enc_spec).permute(0, 2, 1).contiguous()
        #midi_freq = [batch_size*n_frame, n_note, hid_dim] (8*128, 88, 256)
        #print('Decoder_linear_SAtime(1) midi_freq: '+str(midi_freq.shape))

        ## output(freq)
        output_onset_freq = self.sigmoid(self.fc_onset_freq(midi_freq).reshape([batch_size, self.n_frame, self.n_note]))
        output_offset_freq = self.sigmoid(self.fc_offset_freq(midi_freq).reshape([batch_size, self.n_frame, self.n_note]))
        output_mpe_freq = self.sigmoid(self.fc_mpe_freq(midi_freq).reshape([batch_size, self.n_frame, self.n_note]))
        output_velocity_freq = self.fc_velocity_freq(midi_freq).reshape([batch_size, self.n_frame, self.n_note, self.n_velocity])
        #output_onset_freq = [batch_size, n_frame, n_note] (8, 128, 88)
        #output_offset_freq = [batch_size, n_frame, n_note] (8, 128, 88)
        #output_mpe_freq = [batch_size, n_frame, n_note] (8, 128, 88)
        #output_velocity_freq = [batch_size, n_frame, n_note, n_velocity] (8, 128, 88, 128)
        #print('Decoder_linear_SAtime(2) output_onset_freq: '+str(output_onset_freq.shape))
        #print('Decoder_linear_SAtime(2) output_offset_freq: '+str(output_offset_freq.shape))
        #print('Decoder_linear_SAtime(2) output_mpe_freq: '+str(output_mpe_freq.shape))
        #print('Decoder_linear_SAtime(2) output_velocity_freq: '+str(output_velocity_freq.shape))

        ##
        ## SAtime time(64)
        ##
        #midi_time: [batch_size*n_frame, n_note, hid_dim] -> [batch_size*n_note, n_frame, hid_dim]
        midi_time = midi_freq.reshape([batch_size, self.n_frame, self.n_note, self.hid_dim]).permute(0, 2, 1, 3).contiguous().reshape([batch_size*self.n_note, self.n_frame, self.hid_dim])
        pos_time = torch.arange(0, self.n_frame).unsqueeze(0).repeat(batch_size*self.n_note, 1).to(self.device)
        midi_time = self.dropout((midi_time * self.scale_time) + self.pos_embedding_time(pos_time))
        #pos_time = [batch_size*n_note, n_frame] (8*88, 128)
        #midi_time = [batch_size*n_note, n_frame, hid_dim] (8*88, 128, 256)
        #print('Decoder_linear_SAtime(3) pos_time: '+str(pos_time.shape))
        #print('Decoder_linear_SAtime(3) midi_time: '+str(midi_time.shape))

        for layer_time in self.layers_time:
            midi_time = layer_time(midi_time)
        #midi_time = [batch_size*n_note, n_frame, hid_dim] (8*88, 128, 256)
        #print('Decoder_linear_SAtime(4) midi_time: '+str(midi_time.shape))

        ## output(time)
        output_onset_time = self.sigmoid(self.fc_onset_time(midi_time).reshape([batch_size, self.n_note, self.n_frame]).permute(0, 2, 1).contiguous())
        output_offset_time = self.sigmoid(self.fc_offset_time(midi_time).reshape([batch_size, self.n_note, self.n_frame]).permute(0, 2, 1).contiguous())
        output_mpe_time = self.sigmoid(self.fc_mpe_time(midi_time).reshape([batch_size, self.n_note, self.n_frame]).permute(0, 2, 1).contiguous())
        output_velocity_time = self.fc_velocity_time(midi_time).reshape([batch_size, self.n_note, self.n_frame, self.n_velocity]).permute(0, 2, 1, 3).contiguous()
        #output_onset_time = [batch_size, n_frame, n_note] (8, 128, 88)
        #output_offset_time = [batch_size, n_frame, n_note] (8, 128, 88)
        #output_mpe_time = [batch_size, n_frame, n_note] (8, 128, 88)
        #output_velocity_time = [batch_size, n_frame, n_note, n_velocity] (8, 128, 88, 128)
        #print('Decoder_linear_SAtime(5) output_onset_time: '+str(output_onset_time.shape))
        #print('Decoder_linear_SAtime(5) output_offset_time: '+str(output_offset_time.shape))
        #print('Decoder_linear_SAtime(5) output_mpe_time: '+str(output_mpe_time.shape))
        #print('Decoder_linear_SAtime(5) output_velocity_time: '+str(output_velocity_time.shape))

        return output_onset_freq, output_offset_freq, output_mpe_freq, output_velocity_freq, output_onset_time, output_offset_time, output_mpe_time, output_velocity_time


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
