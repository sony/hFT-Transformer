#! python

import argparse
import numpy as np

def note2freq(note_number):
    return 440.0 * pow(2.0, (int(note_number) - 69) / 12)

def sec2frame(sec, nframe_in_sec):
    return int(sec * nframe_in_sec + 0.5)

NUM_PITCH=128
if __name__ == '__main__':
    # option
    parser = argparse.ArgumentParser()
    parser.add_argument('-f_list', help='corpus list file')
    parser.add_argument('-d_note', help='note file directory (input)')
    parser.add_argument('-d_ref', help='reference file directory (output)')
    args = parser.parse_args()

    with open(args.f_list, 'r', encoding='utf-8') as f:
        a_fname = f.readlines()

    for k in range(len(a_fname)):
        fname = a_fname[k].rstrip('\n')
        print(fname)

        with open(args.d_note.rstrip('\n')+'/'+fname+'.txt', 'r', encoding='utf-8') as f:
            a_input = f.readlines()

        fo1 = open(args.d_ref.rstrip('\n')+'/'+fname+'.txt', 'w', encoding='utf-8')
        fo2 = open(args.d_ref.rstrip('\n')+'/'+fname+'_velocity.txt', 'w', encoding='utf-8')

        duration = 0.0
        for i in range(1, len(a_input)):
            onset = a_input[i].rstrip('\n').split('\t')[0]
            offset = a_input[i].rstrip('\n').split('\t')[1]
            velocity = a_input[i].rstrip('\n').split('\t')[2]
            pitch = a_input[i].rstrip('\n').split('\t')[3]
            pitch_freq = note2freq(pitch)
            if float(offset) - float(onset) > 0.0:
                # start - end - pitch(float)
                fo1.write(str(onset)+'\t'+str(offset)+'\t'+str(pitch_freq)+'\n')
                # start - end - pitch(int) - velocity(int)
                fo2.write(str(onset)+'\t'+str(offset)+'\t'+str(pitch)+'\t'+str(velocity)+'\n')
            if duration < float(offset):
                duration = float(offset)
        fo1.close()
        fo2.close()

        ## MPE
        # 16ms/10ms
        nframe_16ms = int(duration * 62.5 + 0.5)+1
        nframe_10ms = int(duration * 100 + 0.5)+1
        a_mpe_16ms = np.zeros((nframe_16ms, NUM_PITCH), dtype=np.int)
        a_mpe_10ms = np.zeros((nframe_10ms, NUM_PITCH), dtype=np.int)
        for n in range(1, len(a_input)):
            onset = float(a_input[n].rstrip('\n').split('\t')[0])
            offset = float(a_input[n].rstrip('\n').split('\t')[1])
            pitch = int(a_input[n].rstrip('\n').split('\t')[3])
            # 16ms
            onset_frame = int(onset*62.5+0.5)
            offset_frame = int(offset*62.5+0.5)
            for i in range(onset_frame, offset_frame+1):
                a_mpe_16ms[i][pitch] = 1
            # 10ms
            onset_frame = int(onset*100+0.5)
            offset_frame = int(offset*100+0.5)
            for i in range(onset_frame, offset_frame+1):
                a_mpe_10ms[i][pitch] = 1

        # 16ms
        fo3 = open(args.d_ref.rstrip('\n')+'/'+fname+'_mpe_16ms.txt', 'w', encoding='utf-8')
        for i in range(len(a_mpe_16ms)):
            fo3.write(str(round(i*0.016, 3)))
            for j in range(NUM_PITCH):
                if a_mpe_16ms[i][j] == 1:
                    fo3.write('\t'+str(note2freq(j)))
            fo3.write('\n')
        fo3.close()

        # 10ms
        fo4 = open(args.d_ref.rstrip('\n')+'/'+fname+'_mpe_10ms.txt', 'w', encoding='utf-8')
        for i in range(len(a_mpe_10ms)):
            fo4.write(str(round(i*0.01, 2)))
            for j in range(NUM_PITCH):
                if a_mpe_10ms[i][j] == 1:
                    fo4.write('\t'+str(note2freq(j)))
            fo4.write('\n')
        fo4.close()
