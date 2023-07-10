#! python

import os
import shutil
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
    parser.add_argument('-f_number', help='MAPS number (tsv) file', default='MAPS_number.tsv')
    parser.add_argument('-d_list', help='MAPS LIST directory', default='LIST_MUS')
    parser.add_argument('-data', help='data to be listed(MUS|others|all)', default='MUS')
    args = parser.parse_args()

    print('** make_list_maps **')
    print(' MAPS number file    : '+str(args.f_number))
    print(' MAPS LIST directory : '+str(args.d_list))
    print(' data type           : '+str(args.data))

    # LIST
    f_test  = open(args.d_list.rstrip('/')+'/test.list', 'w', encoding='utf-8')
    f_train = open(args.d_list.rstrip('/')+'/train.list', 'w', encoding='utf-8')
    f_valid = open(args.d_list.rstrip('/')+'/valid.list', 'w', encoding='utf-8')

    with open(args.f_number, 'r', encoding='utf-8') as fi:
        a_input = fi.readlines()

    # all
    if args.data == 'all':
        for i in range(len(a_input)):
            idx = a_input[i].rstrip('\n').split('\t')[0]
            if (i % 10) == 1:
                f_test.write(idx+'\n')
            elif (i % 10) == 2:
                f_valid.write(idx+'\n')
            else:
                f_train.write(idx+'\n')

    elif args.data == 'MUS':
        # MUS
        a_tune = []
        for i in range(len(a_input)):
            idx = a_input[i].rstrip('\n').split('\t')[0]
            a_data = a_input[i].rstrip('\n').split('\t')[1].split('/')
            code = a_data[1]
            content = a_data[2]
            tune = a_data[len(a_data)-1].rstrip(code).lstrip('MAPS_'+content+'-')
            if (content == 'MUS') and \
               ((code == 'ENSTDkAm') or (code == 'ENSTDkCl')):
                f_test.write(idx+'\n')
                if (tune in a_tune) is False:
                    a_tune.append(tune)

        for i in range(len(a_input)):
            idx = a_input[i].rstrip('\n').split('\t')[0]
            a_data = a_input[i].rstrip('\n').split('\t')[1].split('/')
            code = a_data[1]
            content = a_data[2]
            tune = a_data[len(a_data)-1].rstrip(code).lstrip('MAPS_'+content+'-')
            if (content == 'MUS') and \
               ((code != 'ENSTDkAm') and (code != 'ENSTDkCl')):
                if ((tune in a_tune) is False):
                    f_train.write(idx+'\n')
                else:
                    f_valid.write(idx+'\n')
    else:
        n = 0
        for i in range(len(a_input)):
            idx = a_input[i].rstrip('\n').split('\t')[0]
            a_data = a_input[i].rstrip('\n').split('\t')[1].split('/')
            code = a_data[1]
            content = a_data[2]
            if (content != 'MUS'):
                if (n % 10) == 0:
                    f_test.write(idx+'\n')
                elif (n % 10) == 1:
                    f_valid.write(idx+'\n')
                else:
                    f_train.write(idx+'\n')
                n += 1

    f_test.close()
    f_train.close()
    f_valid.close()

    print('** done **')
