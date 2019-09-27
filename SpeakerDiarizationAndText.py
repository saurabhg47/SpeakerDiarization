"""A demo script showing how to DIARIZATION ON WAV USING UIS-RNN."""
import shutil

import numpy as np
import uisrnn
import librosa
import sys
import xlwt
from ms_excel import *
sys.path.append('ghostvlad')
sys.path.append('visualization')
import toolkits
import model as spkModel
import os

# os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=r"/home/aman/windows downloads/downloads/My First Project-70c19818d95d.json"
# output_filepath = r"/home/aman/windows_desktop/work/Speaker-Diarization-master/wavs/new/" #Final transcript path
# bucketname = "audiofileaman" #Name of the bucket created in the step before
no_speakers = 0
# ===========================================
#        Parse the argument
# ===========================================
import argparse

parser = argparse.ArgumentParser()
# set up training configuration.
parser.add_argument('--gpu', default='', type=str)
parser.add_argument('--resume', default=r'ghostvlad/pretrained/weights.h5', type=str)
parser.add_argument('--data_path', default='4persons', type=str)
# set up network configuration.
parser.add_argument('--net', default='resnet34s', choices=['resnet34s', 'resnet34l'], type=str)
parser.add_argument('--ghost_cluster', default=2, type=int)
parser.add_argument('--vlad_cluster', default=8, type=int)
parser.add_argument('--bottleneck_dim', default=512, type=int)
parser.add_argument('--aggregation_mode', default='gvlad', choices=['avg', 'vlad', 'gvlad'], type=str)
# set up learning rate, training loss and optimizer.
parser.add_argument('--loss', default='softmax', choices=['softmax', 'amsoftmax'], type=str)
parser.add_argument('--test_type', default='normal', choices=['normal', 'hard', 'extend'], type=str)

global args
args = parser.parse_args()

SAVED_MODEL_NAME = 'pretrained/saved_model.uisrnn_benchmark'


def append2dict(speakerSlice, spk_period):
    key = list(spk_period.keys())[0]
    value = list(spk_period.values())[0]
    timeDict = {}
    timeDict['start'] = int(value[0] + 0.5)
    timeDict['stop'] = int(value[1] + 0.5)
    if (key in speakerSlice):
        speakerSlice[key].append(timeDict)
    else:
        speakerSlice[key] = [timeDict]

    return speakerSlice


def arrangeResult(labels,
                  time_spec_rate):  # {'1': [{'start':10, 'stop':20}, {'start':30, 'stop':40}], '2': [{'start':90, 'stop':100}]}
    lastLabel = labels[0]
    speakerSlice = {}
    j = 0
    for i, label in enumerate(labels):
        if (label == lastLabel):
            continue
        speakerSlice = append2dict(speakerSlice, {lastLabel: (time_spec_rate * j, time_spec_rate * i)})
        j = i
        lastLabel = label
    speakerSlice = append2dict(speakerSlice, {lastLabel: (time_spec_rate * j, time_spec_rate * (len(labels)))})
    return speakerSlice


def genMap(intervals):  # interval slices to maptable
    slicelen = [sliced[1] - sliced[0] for sliced in intervals.tolist()]
    mapTable = {}  # vad erased time to origin time, only split points
    idx = 0
    for i, sliced in enumerate(intervals.tolist()):
        mapTable[idx] = sliced[0]
        idx += slicelen[i]
    mapTable[sum(slicelen)] = intervals[-1, -1]

    keys = [k for k, _ in mapTable.items()]
    keys.sort()
    return mapTable, keys


def fmtTime(timeInMillisecond):
    millisecond = timeInMillisecond % 1000
    minute = timeInMillisecond // 1000 // 60
    second = (timeInMillisecond - minute * 60 * 1000) // 1000
    time = '{}:{:02d}.{}'.format(minute, second, millisecond)
    return time


def load_wav(vid_path, sr):
    wav, _ = librosa.load(vid_path, sr=sr)
    intervals = librosa.effects.split(wav, top_db=20)
    wav_output = []
    for sliced in intervals:
        wav_output.extend(wav[sliced[0]:sliced[1]])
    return np.array(wav_output), (intervals / sr * 1000).astype(int)


def lin_spectogram_from_wav(wav, hop_length, win_length, n_fft=1024):
    linear = librosa.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length)  # linear spectrogram
    # print(linear.shape)
    return linear.T


# 0s        1s        2s                  4s                  6s
# |-------------------|-------------------|-------------------|
# |-------------------|
#           |-------------------|
#                     |-------------------|
#                               |-------------------|
def load_data(path, win_length=400, sr=16000, hop_length=160, n_fft=512, embedding_per_second=0.5, overlap_rate=0.5):
    wav, intervals = load_wav(path, sr=sr)
    linear_spect = lin_spectogram_from_wav(wav, hop_length, win_length, n_fft)
    mag, _ = librosa.magphase(linear_spect)  # magnitude
    mag_T = mag.T
    freq, time = mag_T.shape
    spec_mag = mag_T

    spec_len = sr / hop_length / embedding_per_second
    spec_hop_len = spec_len * (1 - overlap_rate)

    cur_slide = 0.0
    utterances_spec = []

    while (True):  # slide window.
        if (cur_slide + spec_len > time):
            break
        spec_mag = mag_T[:, int(cur_slide + 0.5): int(cur_slide + spec_len + 0.5)]

        # preprocessing, subtract mean, divided by time-wise var
        mu = np.mean(spec_mag, 0, keepdims=True)
        std = np.std(spec_mag, 0, keepdims=True)
        spec_mag = (spec_mag - mu) / (std + 1e-5)
        utterances_spec.append(spec_mag)

        cur_slide += spec_hop_len

    return utterances_spec, intervals


def convert_file_to_wav(f_path, new_file):
    print("Begin: convert_file_to_wav")
    cmd = "ffmpeg -i %s -acodec pcm_s16le -ac 1 -ar 16000 %s" % (
        f_path, new_file)
    print("convert_file_to_wav command: %s" % cmd)
    status = False
    try:
        x = os.popen(cmd).read()
        print(x)
        status = True
    except Exception as e:
        print(str(e))
    print("End: convert_file_to_wav")
    return status


def main(file_path, check, embedding_per_second=1.0, overlap_rate=0.5):
    # gpu configuration
    toolkits.initialize_GPU(args)
    if os.path.exists("/tmp/speaker/"):
        shutil.rmtree("/tmp/speaker/")
    os.mkdir("/tmp/speaker/")


    params = {'dim': (257, None, 1),
              'nfft': 512,
              'spec_len': 250,
              'win_length': 400,
              'hop_length': 160,
              'n_classes': 5994,
              'sampling_rate': 16000,
              'normalize': True,
              }
    wav_path_1 = file_path
    if not file_path.endswith(".wav"):
        wav_path_1 = "/tmp/speaker/1.wav"
        convert_file_to_wav(file_path, wav_path_1)
    wav_path_2 = check
    if not check.endswith(".wav"):
        wav_path_2 = "/tmp/speaker/2.wav"
        convert_file_to_wav(check, wav_path_2)

    network_eval = spkModel.vggvox_resnet2d_icassp(input_dim=params['dim'],
                                                   num_class=params['n_classes'],
                                                   mode='eval', args=args)
    network_eval.load_weights(args.resume, by_name=True)

    model_args, _, inference_args = uisrnn.parse_arguments()
    model_args.observation_dim = 512
    uisrnnModel = uisrnn.UISRNN(model_args)
    uisrnnModel.load(SAVED_MODEL_NAME)

    specs, intervals = load_data(wav_path_1, embedding_per_second=embedding_per_second, overlap_rate=overlap_rate)
    mapTable, keys = genMap(intervals)
    if check != '':
        specs1, interval1 = load_data(wav_path_2, embedding_per_second=1.2, overlap_rate=0.4)
        mapTable1, keys1 = genMap(interval1)

    feats = []
    for spec in specs:
        spec = np.expand_dims(np.expand_dims(spec, 0), -1)
        v = network_eval.predict(spec)
        feats += [v]
    featss = np.array(feats)[:, 0, :].astype(float)
    predicted_label = uisrnnModel.predict(featss, inference_args)
    total_speaker = len(set(predicted_label))
    global no_speakers
    # print("predicted_label: %s" % predicted_label)
    no_speakers = len(set(predicted_label))
    # print('total no of speakers', no_speakers)
    time_spec_rate = 1000 * (1.0 / embedding_per_second) * (1.0 - overlap_rate)  # speaker embedding every ?ms
    if check != '':
        for spec1 in specs1:
            spec1 = np.expand_dims(np.expand_dims(spec1, 0), -1)
            v = network_eval.predict(spec1)
            feats += [v]
        featss = np.array(feats)[:, 0, :].astype(float)  # [splits, embedding dim]
        # print("=====================")
        # print(feats)
        # print(featss)
        # print("=====================")
        predicted_label2 = uisrnnModel.predict(featss, inference_args)
        import gc
        print(gc.collect())
        feats = None
        featss = None
        print("After: %s" % gc.collect())
        # print("predicted_label2: %s" % predicted_label2)
        check_speaker = len(set(predicted_label2))

        ms_comparision_status = 'same Speaker' if total_speaker == check_speaker else 'not the same speaker'
        ws.write(iter + 1, 2, ms_comparision_status)

        # print('same Speaker' if total_speaker == check_speaker else 'not the same speaker')
        # print(ms_comparision_status)
        # print('speaker detected as ' + str(predicted_label2[-1]) if total_speaker == check_speaker else '')
    #     speakerSlice2 = arrangeResult(predicted_label2, time_spec_rate)
    #     # print("=============speakerSlice2===============")
    #     for spk, timeDicts in speakerSlice2.items():  # time map to orgin wav(contains mute)
    #         for tid, timeDict in enumerate(timeDicts):
    #             s = 0
    #             e = 0
    #             for i, key in enumerate(keys1):
    #                 if (s != 0 and e != 0):
    #                     break
    #                 if (s == 0 and key > timeDict['start']):
    #                     offset = timeDict['start'] - keys1[i - 1]
    #                     s = mapTable1[keys1[i - 1]] + offset
    #                 if (e == 0 and key > timeDict['stop']):
    #                     offset = timeDict['stop'] - keys1[i - 1]
    #                     e = mapTable1[keys1[i - 1]] + offset
    #
    #             speakerSlice2[spk][tid]['start'] = s
    #             speakerSlice2[spk][tid]['stop'] = e
    #
    #     for spk, timeDicts in speakerSlice2.items():
    #         # print('========= ' + str(spk) + ' =========')
    #         for timeDict in timeDicts:
    #             s = timeDict['start']
    #             e = timeDict['stop']
    #             s = fmtTime(s)  # change point moves to the center of the slice
    #             e = fmtTime(e)
    #             print(s + ' ==> ' + e)
    #     print("=============speakerSlice2===============")
    #     # print(predicted_label,'**************************')
    #     predicted_label2 = None
    # center_duration = int(1000 * (1.0 / embedding_per_second) // 2)
    # speakerSlice = arrangeResult(predicted_label, time_spec_rate)
    #
    # for spk, timeDicts in speakerSlice.items():  # time map to orgin wav(contains mute)
    #     for tid, timeDict in enumerate(timeDicts):
    #         s = 0
    #         e = 0
    #         for i, key in enumerate(keys):
    #             if (s != 0 and e != 0):
    #                 break
    #             if (s == 0 and key > timeDict['start']):
    #                 offset = timeDict['start'] - keys[i - 1]
    #                 s = mapTable[keys[i - 1]] + offset
    #             if (e == 0 and key > timeDict['stop']):
    #                 offset = timeDict['stop'] - keys[i - 1]
    #                 e = mapTable[keys[i - 1]] + offset
    #
    #         speakerSlice[spk][tid]['start'] = s
    #         speakerSlice[spk][tid]['stop'] = e
    #
    # for spk, timeDicts in speakerSlice.items():
    #     # print('========= ' + str(spk) + ' =========')
    #     for timeDict in timeDicts:
    #         s = timeDict['start']
    #         e = timeDict['stop']
    #         s = fmtTime(s)  # change point moves to the center of the slice
    #         e = fmtTime(e)
    #         print(s + ' ==> ' + e)
    #
    # predicted_label = None
if __name__ == '__main__':
    excel_read_obj.excel_read('/home/saurabh/speaker_detection/SpeakerDiarization/new_voice_test.xlsx', 0)
    data = excel_read_obj.details
    tot = len(data)
    print(tot)
    # print("===================")
    wb_Result = xlwt.Workbook()
    ws = wb_Result.add_sheet('Golden_Audio')
    ws.write(0, 0, 'Golden Voice')
    ws.write(0, 1, 'Silver Voice')
    ws.write(0, 2, 'Status')
    ws.write(0, 3, 'Response Time')
    for iter in range(0, tot):
        # print("===================")
        print(iter)
        current_data = data[iter]
        Golden = '/home/saurabh/speaker_detection/SpeakerDiarization/wav/%s' % (current_data.get('GoldenVoice'))
        silver = '/home/saurabh/speaker_detection/SpeakerDiarization/wav/%s' % (current_data.get('SilverVoice'))
        import time
        start_time = time.time()
        # filepath=r'wavs/'
        # filepath=input('enter file path')#r'wavs/rec.wav'
        filepath = Golden
        # verify=int(input('Do you want to verify with another audio 0 for no 1 for yes'))
        verify = 1
        if verify:
            # check=input('Enter audio filepath to check for speaker')
            check = silver
        audio_file_name = filepath.split('/')[-1]
        # print(audio_file_name)
        if verify:
            main(filepath, check, embedding_per_second=1.2, overlap_rate=0.4)
        else:
            main(filepath, check='', embedding_per_second=1.2, overlap_rate=0.4)
        total_response_time = (time.time() - start_time)
        # print("--- %s seconds ---" % (time.time() - start_time))
        #
        # print("===================")
        ws.write(iter+1, 0, current_data.get('GoldenVoice'))
        ws.write(iter + 1, 1, current_data.get('SilverVoice'))
        ws.write(iter + 1, 3, total_response_time)
        wb_Result.save('/home/saurabh/speaker_detection/SpeakerDiarization/new_converted.xls')
# =============================================================================
#     transcript = google_transcribe(audio_file_name)
#     transcript_filename = audio_file_name.split('.')[0] + '.txt'
#     write_transcripts(transcript_filename,transcript)
#
# =============================================================================
