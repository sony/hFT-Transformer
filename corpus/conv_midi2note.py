#! python

import argparse
import json
import mido
#import pretty_midi

'''
def midi2note_pretty_midi(f_midi):
    midi_data = pretty_midi.PrettyMIDI(f_midi)
    a_note = []
    for note in midi_data.instruments[0].notes:
        a_note.append({'onset': note.start,
                       'offset': note.end,
                       'pitch': note.pitch,
                       'velocity': note.velocity})
    #a_note_sort = sorted(sorted(sorted(a_note, key=lambda x: x['pitch']), key=lambda x: x['offset']), key=lambda x: x['onset'])
    a_note_sort = sorted(sorted(a_note, key=lambda x: x['pitch']), key=lambda x: x['onset'])

    return a_note_sort
'''
NUM_PITCH=128
def midi2note(config, f_midi, verbose_flag = False):
    # (1) read MIDI file
    midi_file = mido.MidiFile(f_midi)
    ticks_per_beat = midi_file.ticks_per_beat
    num_tracks = len(midi_file.tracks)

    # (2) tempo curve
    max_ticks_total = 0
    for it in range(len(midi_file.tracks)):
        ticks_total = 0
        for message in midi_file.tracks[it]:
            ticks_total += int(message.time)
        if max_ticks_total < ticks_total:
            max_ticks_total = ticks_total
    a_time_in_sec = [0.0 for i in range(max_ticks_total+1)]
    ticks_curr = 0
    ticks_prev = 0
    tempo_curr = 0
    tempo_prev = 0
    time_in_sec_prev = 0.0
    for im, message in enumerate(midi_file.tracks[0]):
        ticks_curr += message.time
        if 'set_tempo' in str(message):
            tempo_curr = int(message.tempo)
            for i in range(ticks_prev, ticks_curr):
                a_time_in_sec[i] = time_in_sec_prev + ((i-ticks_prev) / ticks_per_beat * tempo_prev / 1e06)
            if ticks_curr > 0:
                time_in_sec_prev = time_in_sec_prev + ((ticks_curr-ticks_prev) / ticks_per_beat * tempo_prev / 1e06)
            tempo_prev = tempo_curr
            ticks_prev = ticks_curr
    for i in range(ticks_prev, max_ticks_total+1):
        a_time_in_sec[i] = time_in_sec_prev + ((i-ticks_prev) / ticks_per_beat * tempo_curr / 1e06)

    # (3) obtain MIDI message
    a_note = []
    a_onset = []
    a_velocity = []
    a_reonset = []
    a_push = []
    a_sustain = []
    for i in range(NUM_PITCH):
        a_onset.append(-1)
        a_velocity.append(-1)
        a_reonset.append(False)
        a_push.append(False)
        a_sustain.append(False)

    ticks = 0
    sustain_flag = False
    for message in midi_file.tracks[num_tracks-1]:
        ticks += message.time
        time_in_sec = a_time_in_sec[ticks]
        if verbose_flag is True:
            #print('[message]'+str(message)+' [ticks]: '+str(ticks/ticks_per_sec))
            print('[message]'+str(message)+' [ticks]: '+str(time_in_sec)+' [time]: '+str(time_in_sec))
        if ('control_change' in str(message)) and ('control=64' in str(message)):
            if message.value < 64:
                # sustain off
                if verbose_flag is True:
                    print('** sustain pedal OFF **')
                for i in range(config['midi']['note_min'], config['midi']['note_max']+1):
                    if (a_push[i] is False) and (a_sustain[i] is True):
                        if verbose_flag is True:
                            print('## output sustain pedal off : '+str(i))
                            print({'onset': a_onset[i],
                                   'offset': time_in_sec,
                                   'pitch': i,
                                   'velocity': a_velocity[i],
                                   'reonset': a_reonset[i]})
                        a_note.append({'onset': a_onset[i],
                                       'offset': time_in_sec,
                                       'pitch': i,
                                       'velocity': a_velocity[i],
                                       'reonset': a_reonset[i]})
                        a_onset[i] = -1
                        a_velocity[i] = -1
                        a_reonset[i] = False
                sustain_flag = False
                for i in range(config['midi']['note_min'], config['midi']['note_max']+1):
                    a_sustain[i] = False
            else:
                # sustain on
                if verbose_flag is True:
                    print('** sustain pedal ON **')
                sustain_flag = True
                for i in range(config['midi']['note_min'], config['midi']['note_max']+1):
                    if a_push[i] is True:
                        a_sustain[i] = True
                        if verbose_flag is True:
                            print('sustain('+str(i)+') ON')
        elif ('note_on' in str(message)) and (int(message.velocity) > 0):
            # note on
            note = message.note
            velocity = message.velocity
            if verbose_flag is True:
                print('++note ON++: '+str(note))
            if (a_push[note] is True) or (a_sustain[note] is True):
                if verbose_flag is True:
                    print('## output reonset : '+str(note))
                    print({'onset': a_onset[note],
                           'offset': time_in_sec,
                           'pitch': note,
                           'velocity': a_velocity[note],
                           'reonset': a_reonset[note]})
                # reonset
                a_note.append({'onset': a_onset[note],
                               'offset': time_in_sec,
                               'pitch': note,
                               'velocity': a_velocity[note],
                               'reonset': a_reonset[note]})
                a_reonset[note] = True
            else:
                a_reonset[note] = False
            a_onset[note] = time_in_sec
            a_velocity[note] = velocity
            a_push[note] = True
            if sustain_flag is True:
                a_sustain[note] = True
                if verbose_flag is True:
                    print('sustain('+str(note)+') ON')
        elif (('note_off' in str(message)) or \
              (('note_on' in str(message)) and (int(message.velocity) == 0))):
            # note off
            note = message.note
            velocity = message.velocity
            if verbose_flag is True:
                print('++note OFF++: '+str(note))
            if (a_push[note] is True) and (a_sustain[note] is False):
                # offset
                if verbose_flag is True:
                    print('## output offset : '+str(note))
                    print({'onset': a_onset[note],
                           'offset': time_in_sec,
                           'pitch': note,
                           'velocity': a_velocity[note],
                           'reonset': a_reonset[note]})
                    print({'onset': a_onset[note],
                           'offset': time_in_sec,
                           'pitch': note,
                           'velocity': a_velocity[note],
                           'reonset': a_reonset[note]})
                a_note.append({'onset': a_onset[note],
                               'offset': time_in_sec,
                               'pitch': note,
                               'velocity': a_velocity[note],
                               'reonset': a_reonset[note]})
                a_onset[note] = -1
                a_velocity[note] = -1
                a_reonset[note] = False
            a_push[note] = False

    for i in range(config['midi']['note_min'], config['midi']['note_max']+1):
        if (a_push[i] is True) or (a_sustain[i] is True):
            if verbose_flag is True:
                print('## output final : '+str(i))
                print({'onset': a_onset[i],
                       'offset': time_in_sec,
                       'pitch': i,
                       'velocity': a_velocity[i],
                       'reonset': a_reonset[i]})
            a_note.append({'onset': a_onset[i],
                           'offset': time_in_sec,
                           'pitch': i,
                           'velocity': a_velocity[i],
                           'reonset': a_reonset[i]})
    a_note_sort = sorted(sorted(a_note, key=lambda x: x['pitch']), key=lambda x: x['onset'])

    return a_note_sort


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d_list', help='corpus list directory')
    parser.add_argument('-d_midi', help='midi file directory (input)')
    parser.add_argument('-d_note', help='note file directory (output)')
    parser.add_argument('-config', help='config file')
    #parser.add_argument('-check', help='double check with pretty_midi', action='store_true')
    args = parser.parse_args()

    print('** conv_midi2note: convert midi to note **')
    print(' directory')
    print('  midi (input)  : '+str(args.d_midi))
    print('  note (output) : '+str(args.d_note))
    print('  corpus list   : '+str(args.d_list))
    print(' config file    : '+str(args.config))

    # read config file
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)

    a_attribute = ['train', 'test', 'valid']
    for attribute in a_attribute:
        print('-'+attribute+'-')
        with open(args.d_list.rstrip('/')+'/'+str(attribute)+'.list', 'r', encoding='utf-8') as f:
            a_input = f.readlines()

        for i in range(len(a_input)):
            fname = a_input[i].rstrip('\n')
            print(fname)

            # convert midi to note
            a_note = midi2note(config, args.d_midi.rstrip('/')+'/'+fname+'.mid', verbose_flag=False)
            '''
            if args.check is True:
                a_note_pretty_midi = midi2note_pretty_midi(args.d_midi.rstrip('/')+'/'+fname+'.mid')
                if len(a_note) != len(a_note_pretty_midi):
                    print('[error] fname: '+str(fname)+' note number mismatch')
                for j in range(len(a_note)):
                    if (a_note[j]['pitch'] != a_note_pretty_midi[j]['pitch']) or \
                       (a_note[j]['velocity'] != a_note_pretty_midi[j]['velocity']) or \
                       (abs(a_note[j]['onset'] - a_note_pretty_midi[j]['onset']) > 0.01):
                        print('[error] fname: '+str(fname)+' note('+str(j)+') data mismatch')
            '''
            with open(args.d_note.rstrip('/')+'/'+fname+'.json', 'w', encoding='utf-8') as f:
                json.dump(a_note, f, ensure_ascii=False, indent=4, sort_keys=False)
            with open(args.d_note.rstrip('/')+'/'+fname+'.txt', 'w', encoding='utf-8') as f:
                f.write('OnsetTime\tOffsetTime\tVelocity\tMidiPitch\n')
                for note in a_note:
                    f.write(str(note['onset'])+'\t')
                    f.write(str(note['offset'])+'\t')
                    f.write(str(note['velocity'])+'\t')
                    f.write(str(note['pitch'])+'\n')

    print('** done **')
