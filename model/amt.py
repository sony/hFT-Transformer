#! python

import pickle
import torch
import numpy as np
import torchaudio
import pretty_midi

class AMT():
    def __init__(self, config, model_path, batch_size=1, verbose_flag=False):
        if verbose_flag is True:
            print('torch version: '+torch.__version__)
            print('torch cuda   : '+str(torch.cuda.is_available()))
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.config = config

        if model_path == None:
            self.model = None
        else:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            self.model = self.model.to(self.device)
            self.model.eval()
            if verbose_flag is True:
                print(self.model)

        self.batch_size = batch_size


    def wav2feature(self, f_wav):
        ### torchaudio
        # torchaudio.transforms.MelSpectrogram()
        # default
        #  sapmle_rate(16000)
        #  win_length(n_fft)
        #  hop_length(win_length//2)
        #  n_fft(400)
        #  f_min(0)
        #  f_max(None)
        #  pad(0)
        #  n_mels(128)
        #  window_fn(hann_window)
        #  center(True)
        #  power(2.0)
        #  pad_mode(reflect)
        #  onesided(True)
        #  norm(None)
        ## melfilter: htk
        ## normalize: none -> slaney

        wave, sr = torchaudio.load(f_wav)
        wave_mono = torch.mean(wave, dim=0)
        tr_fsconv = torchaudio.transforms.Resample(sr, self.config['feature']['sr'])
        wave_mono_16k = tr_fsconv(wave_mono)
        tr_mel = torchaudio.transforms.MelSpectrogram(sample_rate=self.config['feature']['sr'], n_fft=self.config['feature']['fft_bins'], win_length=self.config['feature']['window_length'], hop_length=self.config['feature']['hop_sample'], pad_mode=self.config['feature']['pad_mode'], n_mels=self.config['feature']['mel_bins'], norm='slaney')
        mel_spec = tr_mel(wave_mono_16k)
        a_feature = (torch.log(mel_spec + self.config['feature']['log_offset'])).T

        return a_feature


    def transcript(self, a_feature, mode='combination', ablation_flag=False):
        # a_feature: [num_frame, n_mels]
        a_feature = np.array(a_feature, dtype=np.float32)

        a_tmp_b = np.full([self.config['input']['margin_b'], self.config['feature']['n_bins']], self.config['input']['min_value'], dtype=np.float32)
        len_s = int(np.ceil(a_feature.shape[0] / self.config['input']['num_frame']) * self.config['input']['num_frame']) - a_feature.shape[0]
        a_tmp_f = np.full([len_s+self.config['input']['margin_f'], self.config['feature']['n_bins']], self.config['input']['min_value'], dtype=np.float32)
        a_input = torch.from_numpy(np.concatenate([a_tmp_b, a_feature, a_tmp_f], axis=0))
        # a_input: [margin_b+a_feature.shape[0]+len_s+margin_f, n_bins]

        a_output_onset_A = np.zeros((a_feature.shape[0]+len_s, self.config['midi']['num_note']), dtype=np.float32)
        a_output_offset_A = np.zeros((a_feature.shape[0]+len_s, self.config['midi']['num_note']), dtype=np.float32)
        a_output_mpe_A = np.zeros((a_feature.shape[0]+len_s, self.config['midi']['num_note']), dtype=np.float32)
        a_output_velocity_A = np.zeros((a_feature.shape[0]+len_s, self.config['midi']['num_note']), dtype=np.int8)

        if mode == 'combination':
            a_output_onset_B = np.zeros((a_feature.shape[0]+len_s, self.config['midi']['num_note']), dtype=np.float32)
            a_output_offset_B = np.zeros((a_feature.shape[0]+len_s, self.config['midi']['num_note']), dtype=np.float32)
            a_output_mpe_B = np.zeros((a_feature.shape[0]+len_s, self.config['midi']['num_note']), dtype=np.float32)
            a_output_velocity_B = np.zeros((a_feature.shape[0]+len_s, self.config['midi']['num_note']), dtype=np.int8)

        self.model.eval()
        for i in range(0, a_feature.shape[0], self.config['input']['num_frame']):
            input_spec = (a_input[i:i+self.config['input']['margin_b']+self.config['input']['num_frame']+self.config['input']['margin_f']]).T.unsqueeze(0).to(self.device)

            with torch.no_grad():
                if mode == 'combination':
                    if ablation_flag is True:
                        output_onset_A, output_offset_A, output_mpe_A, output_velocity_A, output_onset_B, output_offset_B, output_mpe_B, output_velocity_B = self.model(input_spec)
                    else:
                        output_onset_A, output_offset_A, output_mpe_A, output_velocity_A, attention, output_onset_B, output_offset_B, output_mpe_B, output_velocity_B = self.model(input_spec)
                    # output_onset: [batch_size, n_frame, n_note]
                    # output_offset: [batch_size, n_frame, n_note]
                    # output_mpe: [batch_size, n_frame, n_note]
                    # output_velocity: [batch_size, n_frame, n_note, n_velocity]
                else:
                    output_onset_A, output_offset_A, output_mpe_A, output_velocity_A = self.model(input_spec)

            a_output_onset_A[i:i+self.config['input']['num_frame']] = (output_onset_A.squeeze(0)).to('cpu').detach().numpy()
            a_output_offset_A[i:i+self.config['input']['num_frame']] = (output_offset_A.squeeze(0)).to('cpu').detach().numpy()
            a_output_mpe_A[i:i+self.config['input']['num_frame']] = (output_mpe_A.squeeze(0)).to('cpu').detach().numpy()
            a_output_velocity_A[i:i+self.config['input']['num_frame']] = (output_velocity_A.squeeze(0).argmax(2)).to('cpu').detach().numpy()

            if mode == 'combination':
                a_output_onset_B[i:i+self.config['input']['num_frame']] = (output_onset_B.squeeze(0)).to('cpu').detach().numpy()
                a_output_offset_B[i:i+self.config['input']['num_frame']] = (output_offset_B.squeeze(0)).to('cpu').detach().numpy()
                a_output_mpe_B[i:i+self.config['input']['num_frame']] = (output_mpe_B.squeeze(0)).to('cpu').detach().numpy()
                a_output_velocity_B[i:i+self.config['input']['num_frame']] = (output_velocity_B.squeeze(0).argmax(2)).to('cpu').detach().numpy()

        if mode == 'combination':
            return a_output_onset_A, a_output_offset_A, a_output_mpe_A, a_output_velocity_A, a_output_onset_B, a_output_offset_B, a_output_mpe_B, a_output_velocity_B
        else:
            return a_output_onset_A, a_output_offset_A, a_output_mpe_A, a_output_velocity_A


    def transcript_stride(self, a_feature, n_offset, mode='combination', ablation_flag=False):
        # a_feature: [num_frame, n_mels]
        a_feature = np.array(a_feature, dtype=np.float32)

        half_frame = int(self.config['input']['num_frame']/2)
        a_tmp_b = np.full([self.config['input']['margin_b'] + n_offset, self.config['feature']['n_bins']], self.config['input']['min_value'], dtype=np.float32)
        tmp_len = a_feature.shape[0] + self.config['input']['margin_b'] + self.config['input']['margin_f'] + half_frame
        len_s = int(np.ceil(tmp_len / half_frame) * half_frame) - tmp_len
        a_tmp_f = np.full([len_s+self.config['input']['margin_f']+(half_frame-n_offset), self.config['feature']['n_bins']], self.config['input']['min_value'], dtype=np.float32)

        a_input = torch.from_numpy(np.concatenate([a_tmp_b, a_feature, a_tmp_f], axis=0))
        # a_input: [n_offset+margin_b+a_feature.shape[0]+len_s+(half_frame-n_offset)+margin_f, n_bins]

        a_output_onset_A = np.zeros((a_feature.shape[0]+len_s, self.config['midi']['num_note']), dtype=np.float32)
        a_output_offset_A = np.zeros((a_feature.shape[0]+len_s, self.config['midi']['num_note']), dtype=np.float32)
        a_output_mpe_A = np.zeros((a_feature.shape[0]+len_s, self.config['midi']['num_note']), dtype=np.float32)
        a_output_velocity_A = np.zeros((a_feature.shape[0]+len_s, self.config['midi']['num_note']), dtype=np.int8)

        if mode == 'combination':
            a_output_onset_B = np.zeros((a_feature.shape[0]+len_s, self.config['midi']['num_note']), dtype=np.float32)
            a_output_offset_B = np.zeros((a_feature.shape[0]+len_s, self.config['midi']['num_note']), dtype=np.float32)
            a_output_mpe_B = np.zeros((a_feature.shape[0]+len_s, self.config['midi']['num_note']), dtype=np.float32)
            a_output_velocity_B = np.zeros((a_feature.shape[0]+len_s, self.config['midi']['num_note']), dtype=np.int8)

        self.model.eval()
        for i in range(0, a_feature.shape[0], half_frame):
            input_spec = (a_input[i:i+self.config['input']['margin_b']+self.config['input']['num_frame']+self.config['input']['margin_f']]).T.unsqueeze(0).to(self.device)

            with torch.no_grad():
                if mode == 'combination':
                    if ablation_flag is True:
                        output_onset_A, output_offset_A, output_mpe_A, output_velocity_A, output_onset_B, output_offset_B, output_mpe_B, output_velocity_B = self.model(input_spec)
                    else:
                        output_onset_A, output_offset_A, output_mpe_A, output_velocity_A, attention, output_onset_B, output_offset_B, output_mpe_B, output_velocity_B = self.model(input_spec)
                    # output_onset: [batch_size, n_frame, n_note]
                    # output_offset: [batch_size, n_frame, n_note]
                    # output_mpe: [batch_size, n_frame, n_note]
                    # output_velocity: [batch_size, n_frame, n_note, n_velocity]
                else:
                    output_onset_A, output_offset_A, output_mpe_A, output_velocity_A = self.model(input_spec)

            a_output_onset_A[i:i+half_frame] = (output_onset_A.squeeze(0)[n_offset:n_offset+half_frame]).to('cpu').detach().numpy()
            a_output_offset_A[i:i+half_frame] = (output_offset_A.squeeze(0)[n_offset:n_offset+half_frame]).to('cpu').detach().numpy()
            a_output_mpe_A[i:i+half_frame] = (output_mpe_A.squeeze(0)[n_offset:n_offset+half_frame]).to('cpu').detach().numpy()
            a_output_velocity_A[i:i+half_frame] = (output_velocity_A.squeeze(0)[n_offset:n_offset+half_frame].argmax(2)).to('cpu').detach().numpy()

            if mode == 'combination':
                a_output_onset_B[i:i+half_frame] = (output_onset_B.squeeze(0)[n_offset:n_offset+half_frame]).to('cpu').detach().numpy()
                a_output_offset_B[i:i+half_frame] = (output_offset_B.squeeze(0)[n_offset:n_offset+half_frame]).to('cpu').detach().numpy()
                a_output_mpe_B[i:i+half_frame] = (output_mpe_B.squeeze(0)[n_offset:n_offset+half_frame]).to('cpu').detach().numpy()
                a_output_velocity_B[i:i+half_frame] = (output_velocity_B.squeeze(0)[n_offset:n_offset+half_frame].argmax(2)).to('cpu').detach().numpy()

        if mode == 'combination':
            return a_output_onset_A, a_output_offset_A, a_output_mpe_A, a_output_velocity_A, a_output_onset_B, a_output_offset_B, a_output_mpe_B, a_output_velocity_B
        else:
            return a_output_onset_A, a_output_offset_A, a_output_mpe_A, a_output_velocity_A


    def mpe2note(self, a_onset=None, a_offset=None, a_mpe=None, a_velocity=None, thred_onset=0.5, thred_offset=0.5, thred_mpe=0.5, mode_velocity='ignore_zero', mode_offset='shorter'):
        ## mode_velocity
        ##  org: 0-127
        ##  ignore_zero: 0-127 (output note does not include 0) (default)

        ## mode_offset
        ##  shorter: use shorter one of mpe and offset (default)
        ##  longer : use longer one of mpe and offset
        ##  offset : use offset (ignore mpe)

        a_note = []
        hop_sec = float(self.config['feature']['hop_sample'] / self.config['feature']['sr'])

        for j in range(self.config['midi']['num_note']):
            # find local maximum
            a_onset_detect = []
            for i in range(len(a_onset)):
                if a_onset[i][j] >= thred_onset:
                    left_flag = True
                    for ii in range(i-1, -1, -1):
                        if a_onset[i][j] > a_onset[ii][j]:
                            left_flag = True
                            break
                        elif a_onset[i][j] < a_onset[ii][j]:
                            left_flag = False
                            break
                    right_flag = True
                    for ii in range(i+1, len(a_onset)):
                        if a_onset[i][j] > a_onset[ii][j]:
                            right_flag = True
                            break
                        elif a_onset[i][j] < a_onset[ii][j]:
                            right_flag = False
                            break
                    if (left_flag is True) and (right_flag is True):
                        if (i == 0) or (i == len(a_onset) - 1):
                            onset_time = i * hop_sec
                        else:
                            if a_onset[i-1][j] == a_onset[i+1][j]:
                                onset_time = i * hop_sec
                            elif a_onset[i-1][j] > a_onset[i+1][j]:
                                onset_time = (i * hop_sec - (hop_sec * 0.5 * (a_onset[i-1][j] - a_onset[i+1][j]) / (a_onset[i][j] - a_onset[i+1][j])))
                            else:
                                onset_time = (i * hop_sec + (hop_sec * 0.5 * (a_onset[i+1][j] - a_onset[i-1][j]) / (a_onset[i][j] - a_onset[i-1][j])))
                        a_onset_detect.append({'loc': i, 'onset_time': onset_time})
            a_offset_detect = []
            for i in range(len(a_offset)):
                if a_offset[i][j] >= thred_offset:
                    left_flag = True
                    for ii in range(i-1, -1, -1):
                        if a_offset[i][j] > a_offset[ii][j]:
                            left_flag = True
                            break
                        elif a_offset[i][j] < a_offset[ii][j]:
                            left_flag = False
                            break
                    right_flag = True
                    for ii in range(i+1, len(a_offset)):
                        if a_offset[i][j] > a_offset[ii][j]:
                            right_flag = True
                            break
                        elif a_offset[i][j] < a_offset[ii][j]:
                            right_flag = False
                            break
                    if (left_flag is True) and (right_flag is True):
                        if (i == 0) or (i == len(a_offset) - 1):
                            offset_time = i * hop_sec
                        else:
                            if a_offset[i-1][j] == a_offset[i+1][j]:
                                offset_time = i * hop_sec
                            elif a_offset[i-1][j] > a_offset[i+1][j]:
                                offset_time = (i * hop_sec - (hop_sec * 0.5 * (a_offset[i-1][j] - a_offset[i+1][j]) / (a_offset[i][j] - a_offset[i+1][j])))
                            else:
                                offset_time = (i * hop_sec + (hop_sec * 0.5 * (a_offset[i+1][j] - a_offset[i-1][j]) / (a_offset[i][j] - a_offset[i-1][j])))
                        a_offset_detect.append({'loc': i, 'offset_time': offset_time})

            time_next = 0.0
            time_offset = 0.0
            time_mpe = 0.0
            for idx_on in range(len(a_onset_detect)):
                # onset
                loc_onset = a_onset_detect[idx_on]['loc']
                time_onset = a_onset_detect[idx_on]['onset_time']

                if idx_on + 1 < len(a_onset_detect):
                    loc_next = a_onset_detect[idx_on+1]['loc']
                    #time_next = loc_next * hop_sec
                    time_next = a_onset_detect[idx_on+1]['onset_time']
                else:
                    loc_next = len(a_mpe)
                    time_next = (loc_next-1) * hop_sec

                # offset
                loc_offset = loc_onset+1
                flag_offset = False
                #time_offset = 0###
                for idx_off in range(len(a_offset_detect)):
                    if loc_onset < a_offset_detect[idx_off]['loc']:
                        loc_offset = a_offset_detect[idx_off]['loc']
                        time_offset = a_offset_detect[idx_off]['offset_time']
                        flag_offset = True
                        break
                if loc_offset > loc_next:
                    loc_offset = loc_next
                    time_offset = time_next

                # offset by MPE
                # (1frame longer)
                loc_mpe = loc_onset+1
                flag_mpe = False
                #time_mpe = 0###
                for ii_mpe in range(loc_onset+1, loc_next):
                    if a_mpe[ii_mpe][j] < thred_mpe:
                        loc_mpe = ii_mpe
                        flag_mpe = True
                        time_mpe = loc_mpe * hop_sec
                        break
                '''
                # (right algorighm)
                loc_mpe = loc_onset
                flag_mpe = False
                for ii_mpe in range(loc_onset+1, loc_next+1):
                    if a_mpe[ii_mpe][j] < thred_mpe:
                        loc_mpe = ii_mpe-1
                        flag_mpe = True
                        time_mpe = loc_mpe * hop_sec
                        break
                '''
                pitch_value = int(j+self.config['midi']['note_min'])
                velocity_value = int(a_velocity[loc_onset][j])

                if (flag_offset is False) and (flag_mpe is False):
                    offset_value = float(time_next)
                elif (flag_offset is True) and (flag_mpe is False):
                    offset_value = float(time_offset)
                elif (flag_offset is False) and (flag_mpe is True):
                    offset_value = float(time_mpe)
                else:
                    if mode_offset == 'offset':
                        ## (a) offset
                        offset_value = float(time_offset)
                    elif mode_offset == 'longer':
                        ## (b) longer
                        if loc_offset >= loc_mpe:
                            offset_value = float(time_offset)
                        else:
                            offset_value = float(time_mpe)
                    else:
                        ## (c) shorter
                        if loc_offset <= loc_mpe:
                            offset_value = float(time_offset)
                        else:
                            offset_value = float(time_mpe)
                if mode_velocity != 'ignore_zero':
                    a_note.append({'pitch': pitch_value, 'onset': float(time_onset), 'offset': offset_value, 'velocity': velocity_value})
                else:
                    if velocity_value > 0:
                        a_note.append({'pitch': pitch_value, 'onset': float(time_onset), 'offset': offset_value, 'velocity': velocity_value})

                if (len(a_note) > 1) and \
                   (a_note[len(a_note)-1]['pitch'] == a_note[len(a_note)-2]['pitch']) and \
                   (a_note[len(a_note)-1]['onset'] < a_note[len(a_note)-2]['offset']):
                    a_note[len(a_note)-2]['offset'] = a_note[len(a_note)-1]['onset']

        a_note = sorted(sorted(a_note, key=lambda x: x['pitch']), key=lambda x: x['onset'])
        return a_note


    def note2midi(self, a_note, f_midi):
        midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)
        for note in a_note:
            instrument.notes.append(pretty_midi.Note(velocity=note['velocity'], pitch=note['pitch'], start=note['onset'], end=note['offset']))
        midi.instruments.append(instrument)
        midi.write(f_midi)

        return
