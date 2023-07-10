#! python

import os
import sys
import argparse

import time

import torch
import torch.nn as nn
import torch.optim as optim

import pickle
import json
import datetime

import train
import dataset
sys.path.append(os.getcwd())
from model.model_spec2midi import *

## model functions
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


## main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', help='config json file', default='config.json')
    parser.add_argument('-d_out', help='parameter directory', default='../output')
    parser.add_argument('-d_dataset', help='dataset directory', default='./dataset')
    parser.add_argument('-n_div_train', help='num of train dataset division(1)', type=int, default=1)
    parser.add_argument('-n_div_valid', help='num of valid dataset division(1)', type=int, default=1)
    parser.add_argument('-n_div_test', help='num of test dataset division(1)', type=int, default=1)
    parser.add_argument('-n_slice', help='dataset slice(0: num_frame, 1>=: this number)(16)', type=int, default=16)
    parser.add_argument('-epoch', help='number of epochs(100)', type=int, default=100)
    parser.add_argument('-resume_epoch', help='number of epoch to resume(-1)', type=int, default=-1)
    parser.add_argument('-resume_div', help='number of div to resume(-1)', type=int, default=-1)
    parser.add_argument('-batch', help='batch size(8)', type=int, default=8)
    parser.add_argument('-lr', help='learning rate(1e-04)', type=float, default=1e-4)
    parser.add_argument('-dropout', help='dropout parameter(0.1)', type=float, default=0.1)
    parser.add_argument('-clip', help='clip parameter(1.0)', type=float, default=1.0)
    parser.add_argument('-seed', type=int, default=1234, help='seed value(1234)')
    parser.add_argument('-cnn_channel', help='number of cnn channel(4)', type=int, default=4)
    parser.add_argument('-cnn_kernel', help='number of cnn kernel(5)', type=int, default=5)
    parser.add_argument('-hid_dim', help='size of hidden layer(256)', type=int, default=256)
    parser.add_argument('-pf_dim', help='size of position-wise feed-forward layer(512)', type=int, default=512)
    parser.add_argument('-enc_layer', help='number of layer of transformer(encoder)(3)', type=int, default=3)
    parser.add_argument('-dec_layer', help='number of layer of transformer(decoder)(3)', type=int, default=3)
    parser.add_argument('-enc_head', help='number of head of transformer(encoder)(4)', type=int, default=4)
    parser.add_argument('-dec_head', help='number of head of transformer(decoder)(4)', type=int, default=4)
    parser.add_argument('-weight_A', help='loss weight for 1st output(1.0)', type=float, default=1.0)
    parser.add_argument('-weight_B', help='loss weight for 2nd output(1.0)', type=float, default=1.0)
    parser.add_argument('-valid_test', help='validation with test data', action='store_true')
    parser.add_argument('-v', help='verbose(print debug)', action='store_true')
    args = parser.parse_args()

    print('** AMT(SPEC2MIDI) training **')
    print(' config file      : '+str(args.config))
    print(' output directory : '+str(args.d_out))
    print(' dataset')
    print('  directory       : '+str(args.d_dataset))
    print('  n_div(train)    : '+str(args.n_div_train))
    print('  n_div(valid)    : '+str(args.n_div_valid))
    print('  n_div(test)     : '+str(args.n_div_test))
    print('  n_slice         : '+str(args.n_slice))
    print(' training parameter')
    print('  epoch           : '+str(args.epoch))
    print('  batch           : '+str(args.batch))
    print('  learning rate   : '+str(args.lr))
    print('  dropout         : '+str(args.dropout))
    print('  clip            : '+str(args.clip))
    print('  seed            : '+str(args.seed))
    print('  validation')
    print('   valid data     : True')
    print('   test data      : '+str(args.valid_test))
    print(' transformer parameter')
    print('  hid_dim         : '+str(args.hid_dim))
    print('  pf_dim          : '+str(args.pf_dim))
    print('  encoder layer   : '+str(args.enc_layer))
    print('          head    : '+str(args.enc_head))
    print('  decoder layer   : '+str(args.dec_layer))
    print('          head    : '+str(args.dec_head))
    print(' CNN parameter')
    print('  channel         : '+str(args.cnn_channel))
    print('  kernel          : '+str(args.cnn_kernel))

    t0 = time.time()

    # (1) read config file
    print('(1) read config file')
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # (2) torch settings
    print('(2) torch settings')
    print(' torch version    : '+torch.__version__)
    print(' torch cuda       : '+str(torch.cuda.is_available()))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    #torch.cuda.set_device(device)

    # (3) network settings
    print('(3) network settings')
    encoder = Encoder_SPEC2MIDI(config['input']['margin_b'],
                                config['input']['num_frame'],
                                config['feature']['n_bins'],
                                args.cnn_channel,
                                args.cnn_kernel,
                                args.hid_dim,
                                args.enc_layer,
                                args.enc_head,
                                args.pf_dim,
                                args.dropout,
                                device)
    decoder = Decoder_SPEC2MIDI(config['input']['num_frame'],
                                config['feature']['n_bins'],
                                config['midi']['num_note'],
                                config['midi']['num_velocity'],
                                args.hid_dim,
                                args.dec_layer,
                                args.dec_head,
                                args.pf_dim,
                                args.dropout,
                                device)
    model = Model_SPEC2MIDI(encoder, decoder)
    model = model.to(device)
    model.apply(initialize_weights);
    print(' The model has {} trainable parameters'.format(count_parameters(model)))

    # (4) training settings
    print('(4) training settings')
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    criterion_onset_A = nn.BCELoss()
    criterion_offset_A = nn.BCELoss()
    criterion_mpe_A = nn.BCELoss()
    criterion_velocity_A = nn.CrossEntropyLoss()

    criterion_onset_B = nn.BCELoss()
    criterion_offset_B = nn.BCELoss()
    criterion_mpe_B = nn.BCELoss()
    criterion_velocity_B = nn.CrossEntropyLoss()

    d_out = args.d_out.rstrip('/')
    if not os.path.exists(d_out):
        os.mkdir(d_out)
    parameters = {
        'config': args.config,
        'parameters': count_parameters(model),
        'd_output': args.d_out,
        'dataset': {
            'd_dataset': args.d_dataset,
            'n_div_train': args.n_div_train,
            'n_div_valid': args.n_div_valid,
            'n_div_test': args.n_div_test,
            'n_slice': args.n_slice
        },
        'training': {
            'epoch': args.epoch,
            'batch': args.batch,
            'lr': args.lr,
            'dropout': args.dropout,
            'clip': args.clip,
            'seed': args.seed,
            'resume_epoch': args.resume_epoch,
            'resume_div': args.resume_div,
            'loss_weight': {
                '1st': args.weight_A,
                '2nd': args.weight_B
            },
            'validation': {
                'test': args.valid_test
            }
        },
        'transformer': {
            'hid_dim': args.hid_dim,
            'pf_dim': args.pf_dim,
            'encoder': {
                'n_layer': args.enc_layer,
                'n_head': args.enc_head
            },
            'decoder': {
                'n_layer': args.dec_layer,
                'n_head': args.dec_head
            }
        },
        'cnn': {
            'channel': args.cnn_channel,
            'kernel': args.cnn_kernel
        }
    }
    with open(d_out+'/parameter.json', 'w', encoding='utf-8') as f:
        json.dump(parameters, f, ensure_ascii=False, indent=4, sort_keys=True)

    epoch_start = 0
    div_start = 0
    best_epoch = 0
    best_div = 0
    best_loss_valid = float('inf')
    a_performance = {
        'loss_train': [],
        'loss_valid': [],
        'loss_test': [],
        'datetime': [],
        'current_epoch': 0,
        'current_div': 0,
        'best_epoch': best_epoch,
        'best_div': best_div,
        'best_loss_valid': best_loss_valid
    }

    # (5) dataset loading (w/o divide)
    print('(5) dataset loading')
    d_dataset = args.d_dataset.rstrip('/')
    if args.n_div_train <= 1:
        dataset_train = dataset.MyDataset(d_dataset+'/feature/train.pkl',
                                          d_dataset+'/label_onset/train.pkl',
                                          d_dataset+'/label_offset/train.pkl',
                                          d_dataset+'/label_mpe/train.pkl',
                                          d_dataset+'/label_velocity/train.pkl',
                                          d_dataset+'/idx/train.pkl',
                                          config,
                                          args.n_slice)
        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch, shuffle=True)
        print('## nstep train: '+str(len(dataloader_train)))
    if args.n_div_valid <= 1:
        dataset_valid = dataset.MyDataset(d_dataset+'/feature/valid.pkl',
                                          d_dataset+'/label_onset/valid.pkl',
                                          d_dataset+'/label_offset/valid.pkl',
                                          d_dataset+'/label_mpe/valid.pkl',
                                          d_dataset+'/label_velocity/valid.pkl',
                                          d_dataset+'/idx/valid.pkl',
                                          config,
                                          args.n_slice)
        dataloader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batch, shuffle=False)
        print('## nstep valid: '+str(len(dataloader_valid)))
    if (args.valid_test is True) and (args.n_div_test <= 1):
        dataset_test = dataset.MyDataset(d_dataset+'/feature/test.pkl',
                                         d_dataset+'/label_onset/test.pkl',
                                         d_dataset+'/label_offset/test.pkl',
                                         d_dataset+'/label_mpe/test.pkl',
                                         d_dataset+'/label_velocity/test.pkl',
                                         d_dataset+'/idx/test.pkl',
                                         config,
                                         args.n_slice)
        dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch, shuffle=False)
        print('## nstep test : '+str(len(dataloader_test)))

    # (6) resume
    if (args.resume_epoch >= 0) or (args.resume_div >= 0):
        if args.resume_epoch < 0:
            args.resume_epoch = 0
        if args.resume_div < 0:
            args.resume_div = 0

        print('(6) resume settings')
        print(' read checkpoint  : model_'+str(args.resume_epoch).zfill(3)+'_'+str(args.resume_div).zfill(3)+'.dat')
        checkpoint = torch.load(d_out+'/model_'+str(args.resume_epoch).zfill(3)+'_'+str(args.resume_div).zfill(3)+'.dat')

        model.load_state_dict(checkpoint['model_dict'])
        #model = checkpoint['model']
        optimizer.load_state_dict(checkpoint['optimizer_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_dict'])
        #random.setstate(checkpoint['random']['random'])
        torch.set_rng_state(checkpoint['random']['torch'])
        torch.random.set_rng_state(checkpoint['random']['torch_random'])
        torch.cuda.set_rng_state(checkpoint['random']['cuda'])
        torch.cuda.torch.cuda.set_rng_state_all(checkpoint['random']['cuda_all'])

        epoch_start = args.resume_epoch
        div_start = args.resume_div + 1
        if div_start >= args.n_div_train:
            epoch_start += 1
            div_start = 0
        best_epoch = int(checkpoint['best_epoch'])
        best_div = int(checkpoint['best_div'])
        best_loss_valid = int(checkpoint['best_loss_valid'])
        current_epoch = checkpoint['epoch']
        current_div = checkpoint['div']

        with open(d_out+'/performance_'+str(args.resume_epoch).zfill(3)+'_'+str(args.resume_div).zfill(3)+'.json', 'r', encoding='utf-8') as f:
            a_performance = json.load(f)
        print(' resume     epoch: '+str(args.resume_epoch)+' div: '+str(args.resume_div))
        print(' checkpoint epoch: '+str(current_epoch)+' div: '+str(current_div))

    # (7) training
    print('(7) training')
    print(' epoch_start      : '+str(epoch_start))
    print(' div_start        : '+str(div_start))
    for epoch in range(epoch_start, args.epoch):
        for div in range(0, args.n_div_train):
            if div < div_start:
                continue
            print('[epoch: '+str(epoch).zfill(3)+' div: '+str(div).zfill(3)+']')

            # (7-1) training
            print('(7-1) training')
            if args.n_div_train > 1:
                dataset_train = dataset.MyDataset(d_dataset+'/feature/train_'+str(div).zfill(3)+'.pkl',
                                                  d_dataset+'/label_onset/train_'+str(div).zfill(3)+'.pkl',
                                                  d_dataset+'/label_offset/train_'+str(div).zfill(3)+'.pkl',
                                                  d_dataset+'/label_mpe/train_'+str(div).zfill(3)+'.pkl',
                                                  d_dataset+'/label_velocity/train_'+str(div).zfill(3)+'.pkl',
                                                  d_dataset+'/idx/train_'+str(div).zfill(3)+'.pkl',
                                                  config,
                                                  args.n_slice)
                dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch, shuffle=True)
                print('## nstep train: '+str(len(dataloader_train)))

            epoch_loss_train = train.train(model, dataloader_train, optimizer,
                                           criterion_onset_A, criterion_offset_A, criterion_mpe_A, criterion_velocity_A,
                                           criterion_onset_B, criterion_offset_B, criterion_mpe_B, criterion_velocity_B,
                                           args.weight_A, args.weight_B,
                                           device, args.v)

            if args.n_div_train > 1:
                del dataset_train, dataloader_train

            # (7-2) validation
            print('(7-2) validation')
            if args.n_div_valid > 1:
                epoch_loss_valid = 0
                num_data_valid = 0
                for div_valid in range(args.n_div_valid):
                    dataset_valid = dataset.MyDataset(d_dataset+'/feature/valid_'+str(div_valid).zfill(3)+'.pkl',
                                                      d_dataset+'/label_onset/valid_'+str(div_valid).zfill(3)+'.pkl',
                                                      d_dataset+'/label_offset/valid_'+str(div_valid).zfill(3)+'.pkl',
                                                      d_dataset+'/label_mpe/valid_'+str(div_valid).zfill(3)+'.pkl',
                                                      d_dataset+'/label_velocity/valid_'+str(div_valid).zfill(3)+'.pkl',
                                                      d_dataset+'/idx/valid_'+str(div_valid).zfill(3)+'.pkl',
                                                      config,
                                                      args.n_slice)
                    dataloader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batch, shuffle=False)
                    print('## nstep valid: '+str(len(dataloader_valid)))
                    retval = train.valid(model, dataloader_valid,
                                         criterion_onset_A, criterion_offset_A, criterion_mpe_A, criterion_velocity_A,
                                         criterion_onset_B, criterion_offset_B, criterion_mpe_B, criterion_velocity_B,
                                         args.weight_A, args.weight_B,
                                         device)
                    epoch_loss_valid += retval[0]
                    num_data_valid += retval[1]
                    del dataset_valid, dataloader_valid
            else:
                epoch_loss_valid, num_data_valid = train.valid(model, dataloader_valid,
                                                               criterion_onset_A, criterion_offset_A, criterion_mpe_A, criterion_velocity_A,
                                                               criterion_onset_B, criterion_offset_B, criterion_mpe_B, criterion_velocity_B,
                                                               args.weight_A, args.weight_B,
                                                               device)
            epoch_loss_valid /= num_data_valid

            # (7-3) test
            if args.valid_test is True:
                print('(7-3) test')
                if args.n_div_test > 1:
                    epoch_loss_test = 0
                    num_data_test = 0
                    for div_test in range(args.n_div_test):
                        dataset_test = dataset.MyDataset(d_dataset+'/feature/test_'+str(div_test).zfill(3)+'.pkl',
                                                         d_dataset+'/label_onset/test_'+str(div_test).zfill(3)+'.pkl',
                                                         d_dataset+'/label_offset/test_'+str(div_test).zfill(3)+'.pkl',
                                                         d_dataset+'/label_mpe/test_'+str(div_test).zfill(3)+'.pkl',
                                                         d_dataset+'/label_velocity/test_'+str(div_test).zfill(3)+'.pkl',
                                                         d_dataset+'/idx/test_'+str(div_test).zfill(3)+'.pkl',
                                                         config,
                                                         args.n_slice)
                        dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch, shuffle=False)
                        print('## nstep test: '+str(len(dataloader_test)))
                        retval = train.valid(model, dataloader_test,
                                             criterion_onset_freq, criterion_offset_freq, criterion_mpe_freq, criterion_velocity_freq,
                                             criterion_onset_time, criterion_offset_time, criterion_mpe_time, criterion_velocity_time,
                                             args.weight_A, args.weight_B,
                                             device)
                        epoch_loss_test += retval[0]
                        num_data_test += retval[1]
                        del dataset_test, dataloader_test
                else:
                    epoch_loss_test, num_data_test = train.valid(model, dataloader_test,
                                                                 criterion_onset_A, criterion_offset_A, criterion_mpe_A, criterion_velocity_A,
                                                                 criterion_onset_B, criterion_offset_B, criterion_mpe_B, criterion_velocity_B,
                                                                 args.weight_A, args.weight_B,
                                                                 device)
                epoch_loss_test /= num_data_test
            else:
                epoch_loss_test = 0.0
            print('[epoch: '+str(epoch).zfill(3)+' div: '+str(div).zfill(3)+']')
            print(' loss(train) :'+str(epoch_loss_train))
            print(' loss(valid) :'+str(epoch_loss_valid))
            if args.valid_test is True:
                print(' loss(test) :'+str(epoch_loss_test))

            # (7-4) save model
            with open(d_out+'/model_'+str(epoch).zfill(3)+'_'+str(div).zfill(3)+'.pkl', 'wb') as f:
                pickle.dump(model, f, protocol=4)
            torch.save({
                'epoch': epoch,
                'div': div,
                'epoch_loss_train': epoch_loss_train,
                'epoch_loss_valid': epoch_loss_valid,
                'epoch_loss_test': epoch_loss_test,
                'best_epoch': epoch,
                'best_div': div,
                'best_loss_valid': best_loss_valid,
                'optimizer_dict': optimizer.state_dict(),
                'scheduler_dict': scheduler.state_dict(),
                'model_dict': model.state_dict(),
                'random': {
                    'torch': torch.get_rng_state(),
                    'torch_random': torch.random.get_rng_state(),
                    'cuda': torch.cuda.get_rng_state(),
                    'cuda_all': torch.cuda.get_rng_state_all()
                },
                'model': model},
                       d_out+'/model_'+str(epoch).zfill(3)+'_'+str(div).zfill(3)+'.dat')

            if best_loss_valid > epoch_loss_valid:
                best_loss_valid = epoch_loss_valid
                best_epoch = epoch
                best_div = div
                with open(d_out+'/best_epoch.txt', 'w') as f:
                    f.write(str(epoch).zfill(3)+'_'+str(div).zfill(3))
                with open(d_out+'/best_model.pkl', 'wb') as f:
                    pickle.dump(model, f, protocol=4)
                torch.save({
                    'epoch': epoch,
                    'div': div,
                    'epoch_loss_train': epoch_loss_train,
                    'epoch_loss_valid': epoch_loss_valid,
                    'epoch_loss_test': epoch_loss_test,
                    'best_epoch': epoch,
                    'best_div': div,
                    'best_loss_valid': best_loss_valid,
                    'optimizer_dict': optimizer.state_dict(),
                    'scheduler_dict': scheduler.state_dict(),
                    'model_dict': model.state_dict(),
                    'random': {
                        'torch': torch.get_rng_state(),
                        'torch_random': torch.random.get_rng_state(),
                        'cuda': torch.cuda.get_rng_state(),
                        'cuda_all': torch.cuda.get_rng_state_all()
                    },
                    'model': model},
                           d_out+'/best_model.dat')

            # (7-5) save performance
            a_performance['loss_train'].append(epoch_loss_train)
            a_performance['loss_valid'].append(epoch_loss_valid)
            a_performance['loss_test'].append(epoch_loss_test)
            a_performance['datetime'].append(datetime.datetime.now().isoformat())
            a_performance['current_epoch'] = epoch
            a_performance['current_div'] = div
            a_performance['best_epoch'] = best_epoch
            a_performance['best_div'] = best_div
            a_performance['best_loss_valid'] = best_loss_valid
            with open(d_out+'/performance.json', 'w', encoding='utf-8') as f:
                json.dump(a_performance, f, ensure_ascii=False, indent=4, sort_keys=True)
            with open(d_out+'/performance_'+str(epoch).zfill(3)+'_'+str(div).zfill(3)+'.json', 'w', encoding='utf-8') as f:
                json.dump(a_performance, f, ensure_ascii=False, indent=4, sort_keys=True)

            # (7-6) scheduler update
            scheduler.step(epoch_loss_valid)

        div_start = 0

    print('** done **')
    t1 = time.time()
    print(' processing time: {0}'.format(t1-t0))
