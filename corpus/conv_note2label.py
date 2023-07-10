#! python

import argparse
import json
import pickle
import numpy as np

def note2label(config, f_note, offset_duration_tolerance_flag):
    # (0) settings
    # tolerance: 50[ms]
    hop_ms = 1000 * config['feature']['hop_sample'] / config['feature']['sr']
    onset_tolerance = int(50.0 / hop_ms + 0.5)
    offset_tolerance = int(50.0 / hop_ms + 0.5)

    with open(f_note, 'r', encoding='utf-8') as f:
        a_note = json.load(f)

    # 62.5 (hop=256, fs=16000)
    nframe_in_sec = config['feature']['sr'] / config['feature']['hop_sample']

    max_offset = 0
    for note in a_note:
        if max_offset < note['offset']:
            max_offset = note['offset']
    
    nframe = int(max_offset * nframe_in_sec + 0.5) + 1
    a_mpe = np.zeros((nframe, config['midi']['num_note']), dtype=np.bool)
    a_onset = np.zeros((nframe, config['midi']['num_note']), dtype=np.float32)
    a_offset = np.zeros((nframe, config['midi']['num_note']), dtype=np.float32)
    a_velocity = np.zeros((nframe, config['midi']['num_note']), dtype=np.int8)

    for i in range(len(a_note)):
        pitch = a_note[i]['pitch'] - config['midi']['note_min']

        # a_note[i]['onset'] in sec
        onset_frame = int(a_note[i]['onset'] * nframe_in_sec + 0.5)
        onset_ms = a_note[i]['onset']*1000.0
        onset_sharpness = onset_tolerance

        # a_note[i]['offset'] in sec
        offset_frame = int(a_note[i]['offset'] * nframe_in_sec + 0.5)
        offset_ms = a_note[i]['offset']*1000.0
        offset_sharpness = offset_tolerance

        if offset_duration_tolerance_flag is True:
            offset_duration_tolerance = int((offset_ms - onset_ms) * 0.2 / hop_ms + 0.5)
            offset_sharpness = max(offset_tolerance, offset_duration_tolerance)

        # velocity
        velocity = a_note[i]['velocity']

        # onset
        for j in range(0, onset_sharpness+1):
            onset_ms_q = (onset_frame + j) * hop_ms
            onset_ms_diff = onset_ms_q - onset_ms
            onset_val = max(0.0, 1.0 - (abs(onset_ms_diff) / (onset_sharpness * hop_ms)))
            if onset_frame+j < nframe:
                a_onset[onset_frame+j][pitch] = max(a_onset[onset_frame+j][pitch], onset_val)
                if (a_onset[onset_frame+j][pitch] >= 0.5):
                    a_velocity[onset_frame+j][pitch] = velocity

        for j in range(1, onset_sharpness+1):
            onset_ms_q = (onset_frame - j) * hop_ms
            onset_ms_diff = onset_ms_q - onset_ms
            onset_val = max(0.0, 1.0 - (abs(onset_ms_diff) / (onset_sharpness * hop_ms)))
            if onset_frame-j >= 0:
                a_onset[onset_frame-j][pitch] = max(a_onset[onset_frame-j][pitch], onset_val)
                if (a_onset[onset_frame-j][pitch] >= 0.5) and (a_velocity[onset_frame-j][pitch] == 0):
                    a_velocity[onset_frame-j][pitch] = velocity

        # mpe
        for j in range(onset_frame, offset_frame+1):
            a_mpe[j][pitch] = 1

        # offset
        offset_flag = True
        for j in range(len(a_note)):
            if a_note[i]['pitch'] != a_note[j]['pitch']:
                continue
            if a_note[i]['offset'] == a_note[j]['onset']:
                offset_flag = False
                break

        if offset_flag is True:
            for j in range(0, offset_sharpness+1):
                offset_ms_q = (offset_frame + j) * hop_ms
                offset_ms_diff = offset_ms_q - offset_ms
                offset_val = max(0.0, 1.0 - (abs(offset_ms_diff) / (offset_sharpness * hop_ms)))
                if offset_frame+j < nframe:
                    a_offset[offset_frame+j][pitch] = max(a_offset[offset_frame+j][pitch], offset_val)
            for j in range(1, offset_sharpness+1):
                offset_ms_q = (offset_frame - j) * hop_ms
                offset_ms_diff = offset_ms_q - offset_ms
                offset_val = max(0.0, 1.0 - (abs(offset_ms_diff) / (offset_sharpness * hop_ms)))
                if offset_frame-j >= 0:
                    a_offset[offset_frame-j][pitch] = max(a_offset[offset_frame-j][pitch],  offset_val)
                    
    # (5-2) output label file
    # mpe        : 0 or 1
    # onset      : 0.0-1.0
    # offset     : 0.0-1.0
    # velocity   : 0 - 127
    a_label = {
        'mpe': a_mpe.tolist(),
        'onset': a_onset.tolist(),
        'offset': a_offset.tolist(),
        'velocity': a_velocity.tolist()
    }

    return a_label

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d_list', help='corpus list directory')
    parser.add_argument('-d_note', help='note file directory (input)')
    parser.add_argument('-d_label', help='label file directory (output)')
    parser.add_argument('-config', help='config file')
    parser.add_argument('-offset_duration_tolerance', help='offset_duration_tolerance ON', action='store_true')
    args = parser.parse_args()

    print('** conv_note2label: convert note to label **')
    print(' directory')
    print('  note (input)  : '+str(args.d_note))
    print('  label (output): '+str(args.d_label))
    print('  corpus list   : '+str(args.d_list))
    print(' config file    : '+str(args.config))
    print(' offset duration tolerance: '+str(args.offset_duration_tolerance))

    # read config file
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)

    a_attribute = ['train', 'test', 'valid']
    for attribute in a_attribute:
        print('-'+attribute+'-')
        with open(args.d_list.rstrip('/')+'/'+str(attribute)+'.list', 'r', encoding='utf-8') as f:
            a_input = f.readlines()

        for i in range(len(a_input)):
            fname = a_input[i].rstrip('\n')
            print(fname)

            # convert note to label
            a_label = note2label(config, args.d_note.rstrip('/')+'/'+fname+'.json', args.offset_duration_tolerance)

            with open(args.d_label.rstrip('/')+'/'+fname+'.pkl', 'wb') as f:
                pickle.dump(a_label, f, protocol=4)

    print('** done **')
