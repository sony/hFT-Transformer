#! python

import argparse

def get_value(data, idx):
    val = ''
    if idx < len(data):
        val = data[idx]
        idx += 1
        if val.count('"') == 1:
            while (idx < len(data)):
                val += data[idx]
                idx += 1
                if '"' in data[idx-1]:
                    break
        val = val.replace('"', '')
    return val, idx

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='input csv file', default='maestro-v3.0.0/maestro-v3.0.0.csv')
    parser.add_argument('-d_list', help='output list directory name', default='LIST')
    args = parser.parse_args()

    print('** make list for MAESTRO **')
    with open(args.i, 'r', encoding='utf-8') as fi:
        a_in = fi.readlines()
    d_list = args.d_list.rstrip('/')

    fo_test = open(d_list+'/test.tsv', 'w', encoding='utf-8')
    fo_train = open(d_list+'/train.tsv', 'w', encoding='utf-8')
    fo_valid = open(d_list+'/valid.tsv', 'w', encoding='utf-8')
    fo_list_test = open(d_list+'/test.list', 'w', encoding='utf-8')
    fo_list_train = open(d_list+'/train.list', 'w', encoding='utf-8')
    fo_list_valid = open(d_list+'/valid.list', 'w', encoding='utf-8')

    fo_test.write('canonical_composer\tcanonical_title\tsplit\tyear\tmidi_filename\taudio_filename\tduration\tnumber\n')
    fo_train.write('canonical_composer\tcanonical_title\tsplit\tyear\tmidi_filename\taudio_filename\tduration\tnumber\n')
    fo_valid.write('canonical_composer\tcanonical_title\tsplit\tyear\tmidi_filename\taudio_filename\tduration\tnumber\n')

    num_train = 0
    num_test = 0
    num_valid = 0

    for i in range(1, len(a_in)):
        data = a_in[i].rstrip('\n').replace('""', '').split(',')
        idx = 0

        composer, idx = get_value(data, idx)
        title, idx = get_value(data, idx)
        split, idx = get_value(data, idx)
        year, idx = get_value(data, idx)
        fname_mid, idx = get_value(data, idx)
        fname_wav, idx = get_value(data, idx)
        duration, idx = get_value(data, idx)
        if split == 'test':
            fo_test.write(composer+'\t'+title+'\t'+split+'\t'+year+'\t'+fname_mid+'\t'+fname_wav+'\t'+duration+'\t')
            fo_test.write(str(num_test).zfill(3)+'\n')
            fo_list_test.write('test_'+str(num_test).zfill(3)+'\n')
            num_test += 1
        elif split == 'train':
            fo_train.write(composer+'\t'+title+'\t'+split+'\t'+year+'\t'+fname_mid+'\t'+fname_wav+'\t'+duration+'\t')
            fo_train.write(str(num_train).zfill(3)+'\n')
            fo_list_train.write('train_'+str(num_train).zfill(3)+'\n')
            num_train += 1
        elif split == 'validation':
            fo_valid.write(composer+'\t'+title+'\t'+split+'\t'+year+'\t'+fname_mid+'\t'+fname_wav+'\t'+duration+'\t')
            fo_valid.write(str(num_valid).zfill(3)+'\n')
            fo_list_valid.write('valid_'+str(num_valid).zfill(3)+'\n')
            num_valid += 1
    fo_test.close()
    fo_train.close()
    fo_valid.close()
    fo_list_test.close()
    fo_list_train.close()
    fo_list_valid.close()
    print('** done **')
