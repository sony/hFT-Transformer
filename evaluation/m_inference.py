#! python

import os
import argparse
import pickle
import json
import sys
sys.path.append(os.getcwd())
from model import amt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f_config', help='config json file', default='../corpus/config.json')
    parser.add_argument('-f_list', help='file list', default='../corpus/MAESTRO-V3/list/test.list')
    parser.add_argument('-d_cp', help='checkpoint directory', default='../checkpoint')
    parser.add_argument('-m', help='input model file', default='best_model.pkl')
    parser.add_argument('-mode', help='mode to transcript (combination|single)', default='combination')
    parser.add_argument('-d_wav', help='corpus wav directory', default='../corpus/MAESTRO-V3/wav')
    parser.add_argument('-d_fe', help='corpus feature directory', default='../corpus/MAESTRO-V3/feature')
    parser.add_argument('-d_mpe', help='output directory for .mpe', default='result/mpe')
    parser.add_argument('-d_note', help='output directory for .json', default='result/note')
    parser.add_argument('-thred_mpe', help='threshold value for mpe detection', type=float, default=0.5)
    parser.add_argument('-thred_onset', help='threshold value for onset detection', type=float, default=0.5)
    parser.add_argument('-thred_offset', help='threshold value for offset detection', type=float, default=0.5)
    parser.add_argument('-calc_feature', help='flag to calculate feature data', action='store_true')
    parser.add_argument('-calc_transcript', help='flag to calculate transcript data', action='store_true')
    parser.add_argument('-n_stride', help='number of samples for offset', type=int, default=0)
    parser.add_argument('-ablation', help='ablation mode', action='store_true')
    args = parser.parse_args()

    print('** AMT: inference for evaluation **')
    print(' file list      : '+str(args.f_list))
    print(' config file    : '+str(args.f_config))
    print(' checkpoint')
    print('  directory     : '+str(args.d_cp))
    print('  model file    : '+str(args.m))
    print(' directories')
    print('  wav           : '+str(args.d_wav))
    print('  feature       : '+str(args.d_fe))
    print('  onset/mpe     : '+str(args.d_mpe))
    print('  json          : '+str(args.d_note))
    print(' threshold value')
    print('  onset         : '+str(args.thred_onset))
    print('  offset        : '+str(args.thred_offset))
    print('  mpe           : '+str(args.thred_mpe))
    print(' calculation')
    print('  wav2feature   : '+str(args.calc_feature))
    print('  transcript    : '+str(args.calc_transcript))
    print(' stride         : '+str(args.n_stride))
    print(' ablation mode  : '+str(args.ablation))

    # parameters
    with open(args.d_cp.rstrip('/')+'/parameter.json', 'r', encoding='utf-8') as f:
        parameters = json.load(f)

    # list file
    a_list = []
    with open(args.f_list, 'r', encoding='utf-8') as f:
        a_list_tmp = f.readlines()
    for fname in a_list_tmp:
        a_list.append(fname.rstrip('\n'))
    del a_list_tmp

    # config file
    with open(args.f_config, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # AMT class
    AMT = amt.AMT(config, args.d_cp.rstrip('/')+'/'+args.m, verbose_flag = False)

    # inference
    out_dir_mpe = args.d_mpe.rstrip('/')
    out_dir_note = args.d_note.rstrip('/')

    for fname in a_list:
        print('['+str(fname)+']')

        # feature
        if args.calc_feature is True:
            a_feature = AMT.wav2feature(args.d_wav.rstrip('/')+'/'+fname+'.wav')
            with open(args.d_fe.rstrip('/')+'/'+fname+'.pkl', 'wb') as f:
                pickle.dump(a_feature, f, protocol=4)
        else:
            with open(args.d_fe.rstrip('/')+'/'+fname+'.pkl', 'rb') as f:
                a_feature = pickle.load(f)

        # transcript
        if args.calc_transcript is True:
            if args.mode == 'combination':
                if args.n_stride > 0:
                    output_1st_onset, output_1st_offset, output_1st_mpe, output_1st_velocity, output_2nd_onset, output_2nd_offset, output_2nd_mpe, output_2nd_velocity = AMT.transcript_stride(a_feature, args.n_stride, mode=args.mode, ablation_flag=args.ablation)
                else:
                    output_1st_onset, output_1st_offset, output_1st_mpe, output_1st_velocity, output_2nd_onset, output_2nd_offset, output_2nd_mpe, output_2nd_velocity = AMT.transcript(a_feature, mode=args.mode, ablation_flag=args.ablation)
            else:
                if args.n_stride > 0:
                    output_1st_onset, output_1st_offset, output_1st_mpe, output_1st_velocity = AMT.transcript_stride(a_feature, args.n_stride, mode=args.mode, ablation_flag=args.ablation)
                else:
                    output_1st_onset, output_1st_offset, output_1st_mpe, output_1st_velocity = AMT.transcript(a_feature, mode=args.mode, ablation_flag=args.ablation)

            with open(out_dir_mpe+'/'+fname+'_1st.onset', 'wb') as f:
                pickle.dump(output_1st_onset, f, protocol=4)
            with open(out_dir_mpe+'/'+fname+'_1st.offset', 'wb') as f:
                pickle.dump(output_1st_offset, f, protocol=4)
            with open(out_dir_mpe+'/'+fname+'_1st.mpe', 'wb') as f:
                pickle.dump(output_1st_mpe, f, protocol=4)
            with open(out_dir_mpe+'/'+fname+'_1st.velocity', 'wb') as f:
                pickle.dump(output_1st_velocity, f, protocol=4)

            if args.mode == 'combination':
                with open(out_dir_mpe+'/'+fname+'_2nd.onset', 'wb') as f:
                    pickle.dump(output_2nd_onset, f, protocol=4)
                with open(out_dir_mpe+'/'+fname+'_2nd.offset', 'wb') as f:
                    pickle.dump(output_2nd_offset, f, protocol=4)
                with open(out_dir_mpe+'/'+fname+'_2nd.mpe', 'wb') as f:
                    pickle.dump(output_2nd_mpe, f, protocol=4)
                with open(out_dir_mpe+'/'+fname+'_2nd.velocity', 'wb') as f:
                    pickle.dump(output_2nd_velocity, f, protocol=4)

        else:
            with open(out_dir_mpe+'/'+fname+'_1st.onset', 'rb') as f:
                output_1st_onset = pickle.load(f)
            with open(out_dir_mpe+'/'+fname+'_1st.offset', 'rb') as f:
                output_1st_offset = pickle.load(f)
            with open(out_dir_mpe+'/'+fname+'_1st.mpe', 'rb') as f:
                output_1st_mpe = pickle.load(f)
            with open(out_dir_mpe+'/'+fname+'_1st.velocity', 'rb') as f:
                output_1st_velocity = pickle.load(f)

            if args.mode == 'combination':
                with open(out_dir_mpe+'/'+fname+'_2nd.onset', 'rb') as f:
                    output_2nd_onset = pickle.load(f)
                with open(out_dir_mpe+'/'+fname+'_2nd.offset', 'rb') as f:
                    output_2nd_offset = pickle.load(f)
                with open(out_dir_mpe+'/'+fname+'_2nd.mpe', 'rb') as f:
                    output_2nd_mpe = pickle.load(f)
                with open(out_dir_mpe+'/'+fname+'_2nd.velocity', 'rb') as f:
                    output_2nd_velocity = pickle.load(f)

        # note (mpe2note)
        a_note_1st_predict = AMT.mpe2note(a_onset=output_1st_onset,
                                          a_offset=output_1st_offset,
                                          a_mpe=output_1st_mpe,
                                          a_velocity=output_1st_velocity,
                                          thred_onset=args.thred_onset,
                                          thred_offset=args.thred_offset,
                                          thred_mpe=args.thred_mpe,
                                          mode_velocity='ignore_zero',
                                          mode_offset='shorter')
        if args.mode == 'combination':
            a_note_2nd_predict = AMT.mpe2note(a_onset=output_2nd_onset,
                                              a_offset=output_2nd_offset,
                                              a_mpe=output_2nd_mpe,
                                              a_velocity=output_2nd_velocity,
                                              thred_onset=args.thred_onset,
                                              thred_offset=args.thred_offset,
                                              thred_mpe=args.thred_mpe,
                                              mode_velocity='ignore_zero',
                                              mode_offset='shorter')
        with open(out_dir_note+'/'+fname+'_1st.json', 'w', encoding='utf-8') as f:
            json.dump(a_note_1st_predict, f, ensure_ascii=False, indent=4, sort_keys=False)
        if args.mode == 'combination':
            with open(out_dir_note+'/'+fname+'_2nd.json', 'w', encoding='utf-8') as f:
                json.dump(a_note_2nd_predict, f, ensure_ascii=False, indent=4, sort_keys=False)

    print('** done **')
