#! python

import argparse
import pickle
import json
import mir_eval
import copy
import numpy as np
import math

def note2freq(note_number):
    return 440.0 * pow(2.0, (int(note_number) - 69) / 12)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f_config', help='config json file', default='../corpus/config.json')
    parser.add_argument('-f_list', help='file list', default='../corpus/MAESTRO/LIST/test.list')
    parser.add_argument('-d_ref', help='reference directory', default='../corpus/MAPS/ref_16ms_new')
    parser.add_argument('-d_est', help='estimation directory', default='result/')
    parser.add_argument('-d_out', help='output directory', default='result/')
    parser.add_argument('-thred_mpe', help='threshold value for MPE (default=0.5)', type=float, default=float(0.5))
    parser.add_argument('-hop', help='hop length(ms) (default=16)', type=int, choices=[10, 16], default=16)
    parser.add_argument('-output', help='output_1st(1st)|output_2nd(2nd)', default='2nd')
    args = parser.parse_args()

    print('** mir_eval: MPE **')
    print(' file list     : '+str(args.f_list))
    print(' config file   : '+str(args.f_config))
    print(' directories')
    print('  reference    : '+str(args.d_ref))
    print('  estimation   : '+str(args.d_est))
    print('  output       : '+str(args.d_out))
    print(' threshold mpe : '+str(args.thred_mpe))
    print(' hop length    : '+str(args.hop))
    print(' output        : '+str(args.output))

    a_list = []
    with open(args.f_list, 'r', encoding='utf-8') as f:
        a_list_tmp = f.readlines()
    for fname in a_list_tmp:
        a_list.append(fname.rstrip('\n'))
    del a_list_tmp
    if args.f_list.endswith('test.list'):
        valid_data = '_test'
    elif args.f_list.endswith('valid.list'):
        valid_data = '_valid'
    elif args.f_list.endswith('train.list'):
        valid_data = '_train'
    else:
        valid_data = ''
    print(' result file   : '+str('result_mpe'+valid_data+'_'+str(args.output)+'.json'))

    with open(args.f_config, 'r', encoding='utf-8') as f:
        config = json.load(f)

    result_tmp = {
        "Precision": 0.0,
        "Recall": 0.0,
        "Accuracy": 0.0,
        "Substitution Error": 0.0,
        "Miss Error": 0.0,
        "False Alarm Error": 0.0,
        "Total Error": 0.0,
        "Chroma Precision": 0.0,
        "Chroma Recall": 0.0,
        "Chroma Accuracy": 0.0,
        "Chroma Substitution Error": 0.0,
        "Chroma Miss Error": 0.0,
        "Chroma False Alarm Error": 0.0,
        "Chroma Total Error": 0.0
    }
    result = {
        '10ms': copy.deepcopy(result_tmp),
        '16ms': copy.deepcopy(result_tmp)
    }

    count = 0
    for fname in a_list:
        print(fname)

        # reference file
        ref_10ms_file = args.d_ref.rstrip('/')+'/'+fname+'_mpe_10ms.txt'

        # estimated file
        with open(args.d_est.rstrip('/')+'/'+fname+'_'+str(args.output)+'.mpe', 'rb') as f:
            a_mpe_est = pickle.load(f)
        nbin=a_mpe_est.shape[1]

        if args.hop == 16:
            # convert estimated file from .mpe to .txt
            ref_16ms_file = args.d_ref.rstrip('/')+'/'+fname+'_mpe_16ms.txt'
            with open(ref_16ms_file, 'r', encoding='utf-8') as f:
                a_ref_16ms = f.readlines()
            nframe = min(len(a_ref_16ms), len(a_mpe_est))

            est_16ms_file = args.d_est.rstrip('/')+'/'+fname+'_mpe_16ms_'+str(args.output)+'.txt'
            fo = open(est_16ms_file, 'w', encoding='utf-8')
            for ii in range(nframe):
                fo.write(str(round(ii*0.016, 3)))
                for jj in range(nbin):
                    if a_mpe_est[ii][jj] >= args.thred_mpe:
                        fo.write('\t'+str(note2freq(jj+config['midi']['note_min'])))
                fo.write('\n')
            fo.close()
            del a_mpe_est

            # calculate score (16ms)
            ref_16ms_times, ref_16ms_freqs = mir_eval.io.load_ragged_time_series(ref_16ms_file)
            est_16ms_times, est_16ms_freqs = mir_eval.io.load_ragged_time_series(est_16ms_file)
            scores_16ms = mir_eval.multipitch.evaluate(ref_16ms_times, ref_16ms_freqs, est_16ms_times, est_16ms_freqs)
            # output (16ms)
            with open(args.d_out.rstrip('/')+'/'+fname+'_result_mpe_16ms_'+str(args.output)+'.json', 'w', encoding='utf-8') as f:
                json.dump(scores_16ms, f, ensure_ascii=False, indent=4, sort_keys=False)

            # calculate score (10ms)
            ref_10ms_times, ref_10ms_freqs = mir_eval.io.load_ragged_time_series(ref_10ms_file)
            nframe_10ms = math.ceil(est_16ms_times[len(est_16ms_times)-1] / 0.01 + 1)
            est_10ms_times = np.array([0.0]*nframe_10ms)
            for ii in range(nframe_10ms):
                est_10ms_times[ii] = ii*0.01
            est_10ms_freqs = mir_eval.multipitch.resample_multipitch(est_16ms_times, est_16ms_freqs, est_10ms_times)
            scores_10ms = mir_eval.multipitch.evaluate(ref_10ms_times, ref_10ms_freqs, est_10ms_times, est_10ms_freqs)
            # output (10ms)
            with open(args.d_out.rstrip('/')+'/'+fname+'_result_mpe_10ms_'+str(args.output)+'.json', 'w', encoding='utf-8') as f:
                json.dump(scores_10ms, f, ensure_ascii=False, indent=4, sort_keys=False)

            # total result
            for attr in scores_16ms:
                result['16ms'][attr] += scores_16ms[attr]
        else:
            # convert estimated file from .mpe to .txt
            with open(ref_10ms_file, 'r', encoding='utf-8') as f:
                a_ref_10ms = f.readlines()
            nframe = min(len(a_ref_10ms), len(a_mpe_est))

            est_10ms_file = args.d_est.rstrip('/')+'/'+fname+'_mpe_10ms_'+str(args.output)+'.txt'
            fo = open(est_10ms_file, 'w', encoding='utf-8')
            for ii in range(nframe):
                fo.write(str(round(ii*0.01, 3)))
                for jj in range(nbin):
                    if a_mpe_est[ii][jj] >= args.thred_mpe:
                        fo.write('\t'+str(note2freq(jj+config['midi']['note_min'])))
                fo.write('\n')
            fo.close()
            del a_mpe_est

            # calculate score
            ref_10ms_times, ref_10ms_freqs = mir_eval.io.load_ragged_time_series(ref_10ms_file)
            est_10ms_times, est_10ms_freqs = mir_eval.io.load_ragged_time_series(est_10ms_file)
            scores_10ms = mir_eval.multipitch.evaluate(ref_10ms_times, ref_10ms_freqs, est_10ms_times, est_10ms_freqs)
            # output
            with open(args.d_out.rstrip('/')+'/'+fname+'_result_mpe_10ms_'+str(args.output)+'.json', 'w', encoding='utf-8') as f:
                json.dump(scores_10ms, f, ensure_ascii=False, indent=4, sort_keys=False)

        # total result
        for attr in scores_10ms:
            result['10ms'][attr] += scores_10ms[attr]
        count += 1

    # output (total)
    for attr in result['10ms']:
        if args.hop == 16:
            result['16ms'][attr] /= count
        result['10ms'][attr] /= count

    # f1-score
    if args.hop == 16:
        if (result['16ms']['Precision'] + result['16ms']['Recall']) > 0.0:
            result['16ms']['f1'] = (2.0 * result['16ms']['Precision'] * result['16ms']['Recall']) / (result['16ms']['Precision'] + result['16ms']['Recall'])
        else:
            result['16ms']['f1'] = 0.0
    if (result['10ms']['Precision'] + result['10ms']['Recall']) > 0.0:
        result['10ms']['f1'] = (2.0 * result['10ms']['Precision'] * result['10ms']['Recall']) / (result['10ms']['Precision'] + result['10ms']['Recall'])
    else:
        result['10ms']['f1'] = 0.0

    with open(args.d_out.rstrip('/')+'/result_mpe'+valid_data+'_'+str(args.output)+'.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4, sort_keys=False)
    print(result)
    print('** done **')
