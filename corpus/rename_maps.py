#! python

import os
import sys
import argparse

def file_check(path):
    a_file = {}
    for pathname, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith('.mid'):
                a_file[filename[:-4]] = pathname
    a_sorted_file = sorted(a_file.items())
    return a_sorted_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d_i', help='MAPS original corpus directory (input)', default='/mnt/hdd1/AMT/corpus/MAPS/MAPS')
    parser.add_argument('-d_o', help='MAPS renamed corpus directory (output)', default='/mnt/hdd1/AMT/corpus/MAPS')
    parser.add_argument('-o', help='rename tsv file', default='MAPS_number.tsv')
    args = parser.parse_args()

    a_filename = file_check(args.d_i)
    fo = open(args.o, 'w', encoding='utf-8')
    for i in range(len(a_filename)):
        number = 'maps_'+str(i).zfill(5)
        filename = a_filename[i][1]+'/'+a_filename[i][0]
        fo.write(str(number)+'\t'+str(filename[len(args.d_i.rstrip('/')+'/'):])+'\n')
        os.symlink(filename+'.wav', args.d_o.rstrip('/')+'/wav/'+number+'.wav')
        os.symlink(filename+'.mid', args.d_o.rstrip('/')+'/midi/'+number+'.mid')
    fo.close()
