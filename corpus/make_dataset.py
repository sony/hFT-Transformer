#! python

import os
import sys
import argparse
import numpy as np
import json
import pickle

def make_dataset(filelist, attribute, d_feature, d_label, d_dataset, config, n_div):
    print('-'+str(attribute)+'-')
    div_flag = False
    if n_div > 1:
        div_flag = True

    # make fname list
    a_fname = []
    a_num_frame = []
    a_total_num_frame = []
    a_total_num_frame_idx = []
    for div in range(n_div):
        a_fname.append([])
        a_num_frame.append([])
        a_total_num_frame.append(config['input']['margin_b'])
        a_total_num_frame_idx.append(0)

    with open(filelist, 'r', encoding='utf-8') as f:
        a_fname_all = f.readlines()
    for i in range(len(a_fname_all)):
        fname = a_fname_all[i].rstrip('\n')
        if fname.startswith('#'):
            continue
        div = 0
        if div_flag is True:
            div = i % n_div
        a_fname[div].append(fname)

        # num frame
        with open(d_feature+'/'+fname+'.pkl', 'rb') as f:
            feature_tmp = pickle.load(f)
        num_frame_feature = feature_tmp.shape[0]
        del feature_tmp
        with open(d_label+'/'+fname+'.pkl', 'rb') as f:
            label_tmp = pickle.load(f)
        num_frame_label = len(label_tmp['mpe'])
        del label_tmp

        if num_frame_feature < num_frame_label:
            print('(warning) '+str(fname)+': num_frame_feature('+str(num_frame_feature)+') < num_frame_label('+str(num_frame_label)+')')
        num_frame = max(num_frame_feature, num_frame_label)

        a_num_frame[div].append(num_frame)
        a_total_num_frame[div] += num_frame + config['input']['margin_f'] + config['input']['num_frame'] - 1
        a_total_num_frame_idx[div] += num_frame
    del a_fname_all

    if div_flag is True:
        for div in range(n_div):
            print('total_num_frame['+str(div).zfill(3)+']: '+str(a_total_num_frame[div]))
            print(a_fname[div])
    else:
        print('total_num_frame     : '+str(a_total_num_frame[0]))
        print(a_fname[0])

    ## idx
    print('** idx **')
    for div in range(n_div):
        print('div: '+str(div)+'/'+str(n_div))
        a_dataset_idx = np.zeros(a_total_num_frame_idx[div], dtype=np.int32)

        loc_i = 0
        loc_d = config['input']['margin_b']
        for i in range(len(a_fname[div])):
            print('(idx) '+str(i)+'/'+str(len(a_fname[div]))+': '+str(a_fname[div][i]))

            num_frame = a_num_frame[div][i]
            a_dataset_idx[loc_i:loc_i+num_frame] = np.arange(loc_d, loc_d+num_frame)
            loc_i += num_frame
            loc_d += num_frame + config['input']['margin_f'] + config['input']['num_frame'] - 1

        if div_flag is True:
            fi = open(d_dataset+'/idx/'+attribute+'_'+str(div).zfill(3)+'.pkl', 'wb')
        else:
            fi = open(d_dataset+'/idx/'+attribute+'.pkl', 'wb')

        pickle.dump(a_dataset_idx, fi, protocol=4)
        fi.close()
        del a_dataset_idx

    ## feature
    print('** feature **')
    if config['feature']['log_offset'] > 0.0:
        zero_value = np.log(config['feature']['log_offset'])
    else:
        zero_value = config['feature']['log_offset']
    for div in range(n_div):
        print('div: '+str(div)+'/'+str(n_div))
        if config['input']['max_value'] > 0.0:
            a_dataset_feature = np.zeros([a_total_num_frame[div], config['feature']['mel_bins']], dtype=np.float32)
        else:
            a_dataset_feature = np.full([a_total_num_frame[div], config['feature']['mel_bins']], zero_value, dtype=np.float32)

        loc_d = config['input']['margin_b']
        for i in range(len(a_fname[div])):
            print('(feature) '+str(i)+'/'+str(len(a_fname[div]))+': '+str(a_fname[div][i]))

            num_frame = a_num_frame[div][i]
            with open(d_feature+'/'+a_fname[div][i]+'.pkl', 'rb') as f:
                feature_tmp = pickle.load(f)
            num_frame_feature = feature_tmp.shape[0]
            if config['input']['max_value'] > 0.0:
                a_dataset_feature[loc_d:loc_d+num_frame_feature] = (feature_tmp - config['input']['min_value']) / (config['input']['max_value'] - config['input']['min_value'])
            else:
                a_dataset_feature[loc_d:loc_d+num_frame_feature] = feature_tmp
            del feature_tmp

            loc_d += num_frame + config['input']['margin_f'] + config['input']['num_frame'] - 1

        if div_flag is True:
            ff = open(d_dataset+'/feature/'+attribute+'_'+str(div).zfill(3)+'.pkl', 'wb')
        else:
            ff = open(d_dataset+'/feature/'+attribute+'.pkl', 'wb')

        pickle.dump(a_dataset_feature, ff, protocol=4)
        ff.close()
        del a_dataset_feature

    ## label
    print('** label(mpe) **')
    for div in range(n_div):
        print('div: '+str(div)+'/'+str(n_div))
        a_dataset_label_mpe = np.zeros([a_total_num_frame[div], config['midi']['num_note']], dtype=np.bool)

        loc_d = config['input']['margin_b']
        for i in range(len(a_fname[div])):
            print('(label(mpe)) '+str(i)+'/'+str(len(a_fname[div]))+': '+str(a_fname[div][i]))
            num_frame = a_num_frame[div][i]

            with open(d_label+'/'+a_fname[div][i]+'.pkl', 'rb') as f:
                label_tmp = pickle.load(f)
            num_frame_label = len(label_tmp['mpe'])
            a_dataset_label_mpe[loc_d:loc_d+num_frame_label] = label_tmp['mpe'][:]
            del label_tmp

            loc_d += num_frame + config['input']['margin_f'] + config['input']['num_frame'] - 1

        if div_flag is True:
            fl_mpe = open(d_dataset+'/label_mpe/'+attribute+'_'+str(div).zfill(3)+'.pkl', 'wb')
        else:
            fl_mpe = open(d_dataset+'/label_mpe/'+attribute+'.pkl', 'wb')

        pickle.dump(a_dataset_label_mpe, fl_mpe, protocol=4)
        fl_mpe.close()
        del a_dataset_label_mpe

    print('** label(onset) **')
    for div in range(n_div):
        print('div: '+str(div)+'/'+str(n_div))
        a_dataset_label_onset = np.zeros([a_total_num_frame[div], config['midi']['num_note']], dtype=np.float32)

        loc_d = config['input']['margin_b']
        for i in range(len(a_fname[div])):
            print('(label(onset)) '+str(i)+'/'+str(len(a_fname[div]))+': '+str(a_fname[div][i]))
            num_frame = a_num_frame[div][i]

            with open(d_label+'/'+a_fname[div][i]+'.pkl', 'rb') as f:
                label_tmp = pickle.load(f)
            num_frame_label = len(label_tmp['mpe'])
            a_dataset_label_onset[loc_d:loc_d+num_frame_label] = label_tmp['onset'][:]
            del label_tmp

            loc_d += num_frame + config['input']['margin_f'] + config['input']['num_frame'] - 1

        if div_flag is True:
            fl_onset = open(d_dataset+'/label_onset/'+attribute+'_'+str(div).zfill(3)+'.pkl', 'wb')
        else:
            fl_onset = open(d_dataset+'/label_onset/'+attribute+'.pkl', 'wb')

        pickle.dump(a_dataset_label_onset, fl_onset, protocol=4)
        fl_onset.close()
        del a_dataset_label_onset

    print('** label(offset) **')
    for div in range(n_div):
        print('div: '+str(div)+'/'+str(n_div))
        a_dataset_label_offset = np.zeros([a_total_num_frame[div], config['midi']['num_note']], dtype=np.float32)

        loc_d = config['input']['margin_b']
        for i in range(len(a_fname[div])):
            print('(label(offset)) '+str(i)+'/'+str(len(a_fname[div]))+': '+str(a_fname[div][i]))
            num_frame = a_num_frame[div][i]

            with open(d_label+'/'+a_fname[div][i]+'.pkl', 'rb') as f:
                label_tmp = pickle.load(f)
            num_frame_label = len(label_tmp['mpe'])
            a_dataset_label_offset[loc_d:loc_d+num_frame_label] = label_tmp['offset'][:]
            del label_tmp

            loc_d += num_frame + config['input']['margin_f'] + config['input']['num_frame'] - 1

        if div_flag is True:
            fl_offset = open(d_dataset+'/label_offset/'+attribute+'_'+str(div).zfill(3)+'.pkl', 'wb')
        else:
            fl_offset = open(d_dataset+'/label_offset/'+attribute+'.pkl', 'wb')

        pickle.dump(a_dataset_label_offset, fl_offset, protocol=4)
        fl_offset.close()
        del a_dataset_label_offset

    print('** label(velocity) **')
    for div in range(n_div):
        print('div: '+str(div)+'/'+str(n_div))
        a_dataset_label_velocity = np.zeros([a_total_num_frame[div], config['midi']['num_note']], dtype=np.int8)

        loc_d = config['input']['margin_b']
        for i in range(len(a_fname[div])):
            print('(label(velocity)) '+str(i)+'/'+str(len(a_fname[div]))+': '+str(a_fname[div][i]))
            num_frame = a_num_frame[div][i]

            with open(d_label+'/'+a_fname[div][i]+'.pkl', 'rb') as f:
                label_tmp = pickle.load(f)
            num_frame_label = len(label_tmp['mpe'])
            a_dataset_label_velocity[loc_d:loc_d+num_frame_label] = label_tmp['velocity'][:]
            del label_tmp

            loc_d += num_frame + config['input']['margin_f'] + config['input']['num_frame'] - 1

        if div_flag is True:
            fl_velocity = open(d_dataset+'/label_velocity/'+attribute+'_'+str(div).zfill(3)+'.pkl', 'wb')
        else:
            fl_velocity = open(d_dataset+'/label_velocity/'+attribute+'.pkl', 'wb')

        pickle.dump(a_dataset_label_velocity, fl_velocity, protocol=4)
        fl_velocity.close()
        del a_dataset_label_velocity

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d_dataset', help='dataset directory(output)', default='/mnt/hdd1/AMT_EVENT/corpus/MAPS/dataset')
    parser.add_argument('-d_list', help='corpus list directory', default='../corpus/MAESTRO/LIST/')
    parser.add_argument('-d_feature', help='feature file directory', default='../corpus/MAESTRO/fe_10ms/MEL/')
    parser.add_argument('-d_label', help='label file directory', default='../corpus/MAESTRO/label_10ms/')
    parser.add_argument('-f_config_in', help='config json file(input)', default='config.json')
    parser.add_argument('-f_config_out', help='config json file(output)', default='config.json')
    parser.add_argument('-n_div_train', help='number of dataset division (train)', type=int, default=1)
    parser.add_argument('-n_div_valid', help='number of dataset division (valid)', type=int, default=1)
    parser.add_argument('-n_div_test', help='number of dataset division (test)', type=int, default=1)
    parser.add_argument('-max_value', help='max feature value', type=float, default=0.0)

    args = parser.parse_args()
    print('** make_dataset **')
    print(' input directories')
    print('  corpus list       : '+str(args.d_list))
    print('  feature           : '+str(args.d_feature))
    print('  label             : '+str(args.d_label))
    print(' dataset directory  : '+str(args.d_dataset))
    print(' config file')
    print('  input             : '+str(args.f_config_in))
    print('  output            : '+str(args.f_config_out))
    print(' number of division')
    print('  train             : '+str(args.n_div_train))
    print('  valid             : '+str(args.n_div_valid))
    print('  test              : '+str(args.n_div_test))

    # read config file
    with open(args.f_config_in, 'r', encoding='utf-8') as f:
        config = json.load(f)

    config['input']['max_value'] = args.max_value
    if config['feature']['log_offset'] > 0.0:
        config['input']['min_value'] = np.log(config['feature']['log_offset']).astype(np.float32)
    else:
        config['input']['min_value'] = config['feature']['log_offset']

    d_list = args.d_list.rstrip('/')
    d_feature = args.d_feature.rstrip('/')
    d_label = args.d_label.rstrip('/')
    d_dataset = args.d_dataset.rstrip('/')
    if not os.path.isdir(d_dataset):
        os.makedirs(d_dataset)

    if not os.path.isdir(d_dataset+'/idx'):
        os.makedirs(d_dataset+'/idx')
    if not os.path.isdir(d_dataset+'/feature'):
        os.makedirs(d_dataset+'/feature')
    if not os.path.isdir(d_dataset+'/label_mpe'):
        os.makedirs(d_dataset+'/label_mpe')
    if not os.path.isdir(d_dataset+'/label_onset'):
        os.makedirs(d_dataset+'/label_onset')
    if not os.path.isdir(d_dataset+'/label_offset'):
        os.makedirs(d_dataset+'/label_offset')
    if not os.path.isdir(d_dataset+'/label_velocity'):
        os.makedirs(d_dataset+'/label_velocity')

    make_dataset(d_list+'/train.list', 'train', d_feature, d_label, d_dataset, config, args.n_div_train)
    make_dataset(d_list+'/valid.list', 'valid', d_feature, d_label, d_dataset, config, args.n_div_valid)
    make_dataset(d_list+'/test.list', 'test', d_feature, d_label, d_dataset, config, args.n_div_test)

    # write config file
    config['input']['min_value'] = float(config['input']['min_value'])
    config['feature']['n_bins'] = config['feature']['mel_bins']
    with open(args.f_config_out, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4, sort_keys=False)

    print('** done **')
