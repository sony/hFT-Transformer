#! python

import torch

##
## train(single)
##
def train_single(model, iterator, optimizer,
                 criterion_onset_A, criterion_offset_A, criterion_mpe_A, criterion_velocity_A,
                 device, verbose_flag):
    model.train()
    epoch_loss = 0
    
    for i, (input_spec, label_onset, label_offset, label_mpe, label_velocity) in enumerate(iterator):
        input_spec = input_spec.to(device, non_blocking=True)
        label_onset = label_onset.to(device, non_blocking=True)
        label_offset = label_offset.to(device, non_blocking=True)
        label_mpe = label_mpe.to(device, non_blocking=True)
        label_velocity = label_velocity.to(device, non_blocking=True)
        # input_spec: [batch_size, n_bins, margin_b+n_frame+margin_f] (8, 256, 192)
        # label_onset: [batch_size, n_frame, n_note] (8, 128, 88)
        # label_velocity: [batch_size, n_frame, n_note] (8, 128, 88)
        if verbose_flag is True:
            print('***** train i : '+str(i)+' *****')
            print('(1) input_spec  : '+str(input_spec.size()))
            print(input_spec)
            print('(1) label_onset   : '+str(label_onset.size()))
            print(label_onset)
            print('(1) label_velocity : '+str(label_velocity.size()))
            print(label_velocity)

        optimizer.zero_grad()
        output_onset_A, output_offset_A, output_mpe_A, output_velocity_A = model(input_spec)
        # output_onset_A: [batch_size, n_frame, n_note] (8, 128, 88)
        # output_velocity_A: [batch_size, n_frame, n_note, n_velocity] (8, 128, 88, 128)
        if verbose_flag is True:
            print('(2) output_onset_A : '+str(output_onset_A.size()))
            print(output_onset_A)
            print('(2) output_velocity_A : '+str(output_velocity_A.size()))
            print(output_velocity_A)

        output_onset_A = output_onset_A.contiguous().view(-1)
        output_offset_A = output_offset_A.contiguous().view(-1)
        output_mpe_A = output_mpe_A.contiguous().view(-1)
        output_velocity_A_dim = output_velocity_A.shape[-1]
        output_velocity_A = output_velocity_A.contiguous().view(-1, output_velocity_A_dim)
        # output_onset_A: [batch_size * n_frame * n_note] (90112)
        # output_velocity_A: [batch_size * n_frame * n_note, n_velocity] (90112, 128)
        if verbose_flag is True:
            print('(3) output_onset_A   : '+str(output_onset_A.size()))
            print('(3) output_velocity_A   : '+str(output_velocity_A.size()))

        label_onset = label_onset.contiguous().view(-1)
        label_offset = label_offset.contiguous().view(-1)
        label_mpe = label_mpe.contiguous().view(-1)
        label_velocity = label_velocity.contiguous().view(-1)
        # label_onset: [batch_size * n_frame * n_note] (90112)
        # label_velocity: [batch_size * n_frame * n_note] (90112)
        if verbose_flag is True:
            print('(4) label_onset   :'+str(label_onset.size()))
            print(label_onset)
            print('(4) label_velocity   :'+str(label_velocity.size()))
            print(label_velocity)

        loss_onset = criterion_onset_A(output_onset_A, label_onset)
        loss_offset = criterion_offset_A(output_offset_A, label_offset)
        loss_mpe = criterion_mpe_A(output_mpe_A, label_mpe)
        loss_velocity = criterion_velocity_A(output_velocity_A, label_velocity)
        loss = loss_onset + loss_offset + loss_mpe + loss_velocity
        if verbose_flag is True:
            print('(5) loss:'+str(loss.size()))
            print(loss)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


##
## validation(single)
##
def valid_single(model, iterator,
                 criterion_onset_A, criterion_offset_A, criterion_mpe_A, criterion_velocity_A,
                 device):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, (input_spec, label_onset, label_offset, label_mpe, label_velocity) in enumerate(iterator):
            input_spec = input_spec.to(device, non_blocking=True)
            label_onset = label_onset.to(device, non_blocking=True)
            label_offset = label_offset.to(device, non_blocking=True)
            label_mpe = label_mpe.to(device, non_blocking=True)
            label_velocity = label_velocity.to(device, non_blocking=True)

            output_onset_A, output_offset_A, output_mpe_A, output_velocity_A = model(input_spec)

            output_onset_A = output_onset_A.contiguous().view(-1)
            output_offset_A = output_offset_A.contiguous().view(-1)
            output_mpe_A = output_mpe_A.contiguous().view(-1)
            output_velocity_A_dim = output_velocity_A.shape[-1]
            output_velocity_A = output_velocity_A.contiguous().view(-1, output_velocity_A_dim)

            label_onset = label_onset.contiguous().view(-1)
            label_offset = label_offset.contiguous().view(-1)
            label_mpe = label_mpe.contiguous().view(-1)
            label_velocity = label_velocity.contiguous().view(-1)

            loss_onset = criterion_onset_A(output_onset_A, label_onset)
            loss_offset = criterion_offset_A(output_offset_A, label_offset)
            loss_mpe = criterion_mpe_A(output_mpe_A, label_mpe)
            loss_velocity = criterion_velocity_A(output_velocity_A, label_velocity)
            loss = loss_onset + loss_offset + loss_mpe + loss_velocity

            epoch_loss += loss.item()

    return epoch_loss, len(iterator)


##
## train(combination)
##
def train_combination(model, iterator, optimizer,
                      criterion_onset_A, criterion_offset_A, criterion_mpe_A, criterion_velocity_A,
                      criterion_onset_B, criterion_offset_B, criterion_mpe_B, criterion_velocity_B,
                      weight_A, weight_B,
                      device, verbose_flag):
    model.train()
    epoch_loss = 0
    
    for i, (input_spec, label_onset, label_offset, label_mpe, label_velocity) in enumerate(iterator):
        input_spec = input_spec.to(device, non_blocking=True)
        label_onset = label_onset.to(device, non_blocking=True)
        label_offset = label_offset.to(device, non_blocking=True)
        label_mpe = label_mpe.to(device, non_blocking=True)
        label_velocity = label_velocity.to(device, non_blocking=True)
        # input_spec: [batch_size, n_bins, margin_b+n_frame+margin_f] (8, 256, 192)
        # label_onset: [batch_size, n_frame, n_note] (8, 128, 88)
        # label_velocity: [batch_size, n_frame, n_note] (8, 128, 88)
        if verbose_flag is True:
            print('***** train i : '+str(i)+' *****')
            print('(1) input_spec  : '+str(input_spec.size()))
            print(input_spec)
            print('(1) label_mpe   : '+str(label_mpe.size()))
            print(label_mpe)
            print('(1) label_velocity : '+str(label_velocity.size()))
            print(label_velocity)

        optimizer.zero_grad()
        output_onset_A, output_offset_A, output_mpe_A, output_velocity_A, output_onset_B, output_offset_B, output_mpe_B, output_velocity_B = model(input_spec)
        # output_onset_A: [batch_size, n_frame, n_note] (8, 128, 88)
        # output_onset_B: [batch_size, n_frame, n_note] (8, 128, 88)
        # output_velocity_A: [batch_size, n_frame, n_note, n_velocity] (8, 128, 88, 128)
        # output_velocity_B: [batch_size, n_frame, n_note, n_velocity] (8, 128, 88, 128)

        if verbose_flag is True:
            print('(2) output_onset_A : '+str(output_onset_A.size()))
            print(output_onset_A)
            print('(2) output_onset_B : '+str(output_onset_B.size()))
            print(output_onset_B)
            print('(2) output_velocity_A : '+str(output_velocity_A.size()))
            print(output_velocity_A)
            print('(2) output_velocity_B : '+str(output_velocity_B.size()))
            print(output_velocity_B)

        output_onset_A = output_onset_A.contiguous().view(-1)
        output_offset_A = output_offset_A.contiguous().view(-1)
        output_mpe_A = output_mpe_A.contiguous().view(-1)
        output_velocity_A_dim = output_velocity_A.shape[-1]
        output_velocity_A = output_velocity_A.contiguous().view(-1, output_velocity_A_dim)

        output_onset_B = output_onset_B.contiguous().view(-1)
        output_offset_B = output_offset_B.contiguous().view(-1)
        output_mpe_B = output_mpe_B.contiguous().view(-1)
        output_velocity_B_dim = output_velocity_B.shape[-1]
        output_velocity_B = output_velocity_B.contiguous().view(-1, output_velocity_B_dim)

        # output_onset_A: [batch_size * n_frame * n_note] (90112)
        # output_onset_B: [batch_size * n_frame * n_note] (90112)
        # output_velocity_A: [batch_size * n_note * n_frame, n_velocity] (90112, 128)
        # output_velocity_B: [batch_size * n_note * n_frame, n_velocity] (90112, 128)

        if verbose_flag is True:
            print('(3) output_onset_A : '+str(output_onset_A.size()))
            print('(3) output_onset_B : '+str(output_onset_B.size()))
            print('(3) output_velocity_A : '+str(output_velocity_A.size()))
            print('(3) output_velocity_B : '+str(output_velocity_B.size()))

        label_onset = label_onset.contiguous().view(-1)
        label_offset = label_offset.contiguous().view(-1)
        label_mpe = label_mpe.contiguous().view(-1)
        label_velocity = label_velocity.contiguous().view(-1)
        # label_onset: [batch_size * n_frame * n_note] (90112)
        # label_velocity: [batch_size * n_frame * n_note] (90112)
        if verbose_flag is True:
            print('(4) label_onset   :'+str(label_onset.size()))
            print(label_onset)
            print('(4) label_velocity   :'+str(label_velocity.size()))
            print(label_velocity)

        loss_onset_A = criterion_onset_A(output_onset_A, label_onset)
        loss_offset_A = criterion_offset_A(output_offset_A, label_offset)
        loss_mpe_A = criterion_mpe_A(output_mpe_A, label_mpe)
        loss_velocity_A = criterion_velocity_A(output_velocity_A, label_velocity)
        loss_A = loss_onset_A + loss_offset_A + loss_mpe_A + loss_velocity_A

        loss_onset_B = criterion_onset_B(output_onset_B, label_onset)
        loss_offset_B = criterion_offset_B(output_offset_B, label_offset)
        loss_mpe_B = criterion_mpe_B(output_mpe_B, label_mpe)
        loss_velocity_B = criterion_velocity_B(output_velocity_B, label_velocity)
        loss_B = loss_onset_B + loss_offset_B + loss_mpe_B + loss_velocity_B

        loss = weight_A * loss_A + weight_B * loss_B
        if verbose_flag is True:
            print('(5) loss:'+str(loss.size()))
            print(loss)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


##
## validation(combination)
##
def valid_combination(model, iterator,
                      criterion_onset_A, criterion_offset_A, criterion_mpe_A, criterion_velocity_A,
                      criterion_onset_B, criterion_offset_B, criterion_mpe_B, criterion_velocity_B,
                      weight_A, weight_B,
                      device):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for i, (input_spec, label_onset, label_offset, label_mpe, label_velocity) in enumerate(iterator):
            input_spec = input_spec.to(device, non_blocking=True)
            label_onset = label_onset.to(device, non_blocking=True)
            label_offset = label_offset.to(device, non_blocking=True)
            label_mpe = label_mpe.to(device, non_blocking=True)
            label_velocity = label_velocity.to(device, non_blocking=True)

            output_onset_A, output_offset_A, output_mpe_A, output_velocity_A, output_onset_B, output_offset_B, output_mpe_B, output_velocity_B = model(input_spec)

            output_onset_A = output_onset_A.contiguous().view(-1)
            output_offset_A = output_offset_A.contiguous().view(-1)
            output_mpe_A = output_mpe_A.contiguous().view(-1)
            output_velocity_A_dim = output_velocity_A.shape[-1]
            output_velocity_A = output_velocity_A.contiguous().view(-1, output_velocity_A_dim)

            output_onset_B = output_onset_B.contiguous().view(-1)
            output_offset_B = output_offset_B.contiguous().view(-1)
            output_mpe_B = output_mpe_B.contiguous().view(-1)
            output_velocity_B_dim = output_velocity_B.shape[-1]
            output_velocity_B = output_velocity_B.contiguous().view(-1, output_velocity_B_dim)

            label_onset = label_onset.contiguous().view(-1)
            label_offset = label_offset.contiguous().view(-1)
            label_mpe = label_mpe.contiguous().view(-1)
            label_velocity = label_velocity.contiguous().view(-1)

            loss_onset_A = criterion_onset_A(output_onset_A, label_onset)
            loss_offset_A = criterion_offset_A(output_offset_A, label_offset)
            loss_mpe_A = criterion_mpe_A(output_mpe_A, label_mpe)
            loss_velocity_A = criterion_velocity_A(output_velocity_A, label_velocity)
            loss_A = loss_onset_A + loss_offset_A + loss_mpe_A + loss_velocity_A

            loss_onset_B = criterion_onset_B(output_onset_B, label_onset)
            loss_offset_B = criterion_offset_B(output_offset_B, label_offset)
            loss_mpe_B = criterion_mpe_B(output_mpe_B, label_mpe)
            loss_velocity_B = criterion_velocity_B(output_velocity_B, label_velocity)
            loss_B = loss_onset_B + loss_offset_B + loss_mpe_B + loss_velocity_B

            loss = weight_A * loss_A + weight_B * loss_B

            epoch_loss += loss.item()

    return epoch_loss, len(iterator)
