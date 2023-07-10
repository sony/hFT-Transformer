#! python

import argparse
import json
import pickle
import sys
import os
sys.path.append(os.getcwd())
from model import amt


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-d_list', help='corpus list directory')
    parser.add_argument('-d_wav', help='wav file directory (input)')
    parser.add_argument('-d_feature', help='feature file directory (output)')
    parser.add_argument('-config', help='config file')
    args = parser.parse_args()

    print('** conv_wav2fe: convert wav to feature **')
    print(' directory')
    print('  wav     (input) : '+str(args.d_wav))
    print('  feature (output): '+str(args.d_feature))
    print('  corpus list     : '+str(args.d_list))
    print(' config file      : '+str(args.config))

    # read config file
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # AMT class
    AMT = amt.AMT(config, None, None)

    a_attribute = ['train', 'test', 'valid']
    for attribute in a_attribute:
        print('-'+attribute+'-')
        with open(args.d_list.rstrip('/')+'/'+str(attribute)+'.list', 'r', encoding='utf-8') as f:
            a_input = f.readlines()

        for i in range(len(a_input)):
            fname = a_input[i].rstrip('\n')
            print(fname)

            # convert wav to feature
            a_feature = AMT.wav2feature(args.d_wav.rstrip('/')+'/'+fname+'.wav')
            with open(args.d_feature.rstrip('/')+'/'+fname+'.pkl', 'wb') as f:
                pickle.dump(a_feature, f, protocol=4)

    print('** done **')
