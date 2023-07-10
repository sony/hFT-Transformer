#! python

import argparse
import json
import mir_eval
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f_list', help='file list', default='../corpus/MAESTRO/LIST/test.list')
    parser.add_argument('-d_ref', help='reference directory', default='../corpus/MAPS/ref_16ms_new')
    parser.add_argument('-d_est', help='estimation directory', default='result/')
    parser.add_argument('-d_out', help='output directory', default='result/')
    parser.add_argument('-velocity', help='w/ velocity', action='store_true')
    parser.add_argument('-output', help='output_1st(1st)|output_2nd(2nd)', default='2nd')
    args = parser.parse_args()

    print('** mir_eval: transcription (note) **')
    print(' file list     : '+str(args.f_list))
    print(' directories')
    print('  reference    : '+str(args.d_ref))
    print('  estimation   : '+str(args.d_est))
    print('  output       : '+str(args.d_out))
    if args.velocity is True:
        print(' with velocity : ON')
    else:
        print(' with velocity : OFF')
    print(' output        : '+str(args.output))

    def _load_transcription_velocity(filename):
        """Loader for data in the format start, end, pitch, velocity."""
        starts, ends, pitches, velocities = mir_eval.io.load_delimited(
            filename, [float, float, int, int])
        # Stack into an interval matrix
        intervals = np.array([starts, ends]).T
        # return pitches and velocities as np.ndarray
        pitches = np.array(pitches)
        velocities = np.array(velocities)
        return intervals, pitches, velocities

    # list file
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

    if args.velocity is False:
        print(' result file   : '+str('result_note'+valid_data+'_'+str(args.output)+'.json'))
        result = {
            "Precision": 0.0,
            "Recall": 0.0,
            "F-measure": 0.0,
            "Average_Overlap_Ratio": 0.0,
            "Precision_no_offset": 0.0,
            "Recall_no_offset": 0.0,
            "F-measure_no_offset": 0.0,
            "Average_Overlap_Ratio_no_offset": 0.0,
            "Onset_Precision": 0.0,
            "Onset_Recall": 0.0,
            "Onset_F-measure": 0.0,
            "Offset_Precision": 0.0,
            "Offset_Recall": 0.0,
            "Offset_F-measure": 0.0
        }
    else:
        print(' result file   : '+str('result_note_velocity'+valid_data+'_'+str(args.output)+'.json'))
        result = {
            "Precision": 0.0,
            "Recall": 0.0,
            "F-measure": 0.0,
            "Average_Overlap_Ratio": 0.0,
            "Precision_no_offset": 0.0,
            "Recall_no_offset": 0.0,
            "F-measure_no_offset": 0.0,
            "Average_Overlap_Ratio_no_offset": 0.0
        }

    count = 0
    for fname in a_list:
        print(fname)

        # convert estimated file from json to txt
        with open(args.d_est.rstrip('/')+'/'+fname+'_'+str(args.output)+'.json', 'r', encoding='utf-8') as f:
            est_obj = json.load(f)

        est_file = args.d_est.rstrip('/')+'/'+fname+'_'+str(args.output)+'.txt'
        fo = open(est_file, 'w', encoding='utf-8')
        if args.velocity is False:
            for obj in est_obj:
                pitch_freq = 440.0*pow(2.0, (int(obj['pitch']) - 69)/12)
                if obj['offset'] - obj['onset'] > 0.0:
                    fo.write(str(obj['onset'])+'\t'+str(obj['offset'])+'\t'+str(pitch_freq)+'\n')
        else:
            for obj in est_obj:
                if obj['offset'] - obj['onset'] > 0.0:
                    fo.write(str(obj['onset'])+'\t'+str(obj['offset'])+'\t'+str(obj['pitch'])+'\t'+str(obj['velocity'])+'\n')
        fo.close()
        del est_obj

        # calculate score
        if args.velocity is False:
            ref_file = args.d_ref.rstrip('/')+'/'+fname+'.txt'
            out_file = args.d_out.rstrip('/')+'/'+fname+'_result_note_'+str(args.output)+'.json'
            ref_int, ref_pitch = mir_eval.io.load_valued_intervals(ref_file)
            est_int, est_pitch = mir_eval.io.load_valued_intervals(est_file)
            scores = mir_eval.transcription.evaluate(ref_int, ref_pitch, est_int, est_pitch)
        else:
            ref_file = args.d_ref.rstrip('/')+'/'+fname+'_velocity.txt'
            out_file = args.d_out.rstrip('/')+'/'+fname+'_result_note_velocity_'+str(args.output)+'.json'
            ref_int, ref_pitch, ref_vel = _load_transcription_velocity(ref_file)
            est_int, est_pitch, est_vel = _load_transcription_velocity(est_file)
            scores = mir_eval.transcription_velocity.evaluate(ref_int, ref_pitch, ref_vel,
                                                              est_int, est_pitch, est_vel)

        # output
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(scores, f, ensure_ascii=False, indent=4, sort_keys=False)

        # total result
        for attr in scores:
            result[attr] += scores[attr]
        count += 1

    # output (total)
    for attr in result:
        result[attr] /= count
    if args.velocity is False:
        fo = open(args.d_est.rstrip('/')+'/result_note'+valid_data+'_'+str(args.output)+'.json', 'w', encoding='utf-8')
    else:
        fo = open(args.d_est.rstrip('/')+'/result_note_velocity'+valid_data+'_'+str(args.output)+'.json', 'w', encoding='utf-8')
    json.dump(result, fo, ensure_ascii=False, indent=4, sort_keys=False)
    fo.close()
    print(result)
    print('** done **')
