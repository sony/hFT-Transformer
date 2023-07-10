#! python

import torch
import pickle

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, f_feature, f_label_onset, f_label_offset, f_label_mpe, f_label_velocity, f_idx, config, n_slice):
        super().__init__()

        with open(f_feature, 'rb') as f:
            feature = pickle.load(f)

        with open(f_label_onset, 'rb') as f:
            label_onset = pickle.load(f)
        with open(f_label_offset, 'rb') as f:
            label_offset = pickle.load(f)
        with open(f_label_mpe, 'rb') as f:
            label_mpe = pickle.load(f)
        if f_label_velocity is not None:
            self.flag_velocity = True
            with open(f_label_velocity, 'rb') as f:
                label_velocity = pickle.load(f)
        else:
            self.flag_velocity = False

        with open(f_idx, 'rb') as f:
            idx = pickle.load(f)

        self.feature = torch.from_numpy(feature)
        self.label_onset = torch.from_numpy(label_onset)
        self.label_offset = torch.from_numpy(label_offset)
        self.label_mpe = torch.from_numpy(label_mpe)
        if self.flag_velocity:
            self.label_velocity = torch.from_numpy(label_velocity)
        if n_slice > 1:
            idx_tmp = torch.from_numpy(idx)
            self.idx = idx_tmp[:int(len(idx_tmp) / n_slice) * n_slice][::n_slice]
        else:
            self.idx = torch.from_numpy(idx)
        self.config = config
        self.data_size = len(self.idx)

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        # margin: 32
        # num_frame: 128
        idx_feature_s = self.idx[idx] - self.config['input']['margin_b']
        idx_feature_e = self.idx[idx] + self.config['input']['num_frame'] + self.config['input']['margin_f']

        idx_label_s = self.idx[idx]
        idx_label_e = self.idx[idx] + self.config['input']['num_frame']

        # a_feature: [margin+num_frame+margin, n_feature] -(transpose)-> spec: [n_feature, margin+num_frame+margin]
        spec = (self.feature[idx_feature_s:idx_feature_e]).T

        # label_onset: [num_frame, n_note]
        label_onset = self.label_onset[idx_label_s:idx_label_e]

        # label_offset: [num_frame, n_note]
        label_offset = self.label_offset[idx_label_s:idx_label_e]

        # label_mpe: [num_frame, n_note]
        # bool -> float
        label_mpe = self.label_mpe[idx_label_s:idx_label_e].float()

        # label_velocity: [num_frame, n_note]
        # int8 -> long
        if self.flag_velocity:
            label_velocity = self.label_velocity[idx_label_s:idx_label_e].long()
            return spec, label_onset, label_offset, label_mpe, label_velocity
        else:
            return spec, label_onset, label_offset, label_mpe
