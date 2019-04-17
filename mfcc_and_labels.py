import file_import as fi
import numpy as np
from python_speech_features import mfcc
import scipy.io.wavfile as wav_reader
from collections import Counter


# 该文件主要定义音频mfcc获取方法与文字向量转化方法等


# 参数分别为 音频地址，取多少个特征，上下文个数
def get_audio_mfcc(audio_file_path, numcep, num_context):
    fs, data = wav_reader.read(audio_file_path)
    origin_inputs = mfcc(data, samplerate=fs, numcep=numcep)  # 返回为分段数*numce（提取特征个数）
    origin_inputs = origin_inputs[::2]  # 神网模型为双向rnn 所以隔一个取一个
    train_inputs = []
    train_inputs = np.array(train_inputs)
    train_inputs.resize([origin_inputs.shape[0], (1 + num_context * 2) * numcep])
    # 开始补0和填充train_input数组
    need_add_zero_max = origin_inputs.shape[0] - num_context
    time_slices = range(train_inputs.shape[0])
    empty_mfcc = list(0. for _ in range(numcep))
    for time_slice in time_slices:
        add_empty_pre = []
        add_empty_pos = []
        # 前序补0
        if time_slice < num_context:
            add_empty_pre = empty_mfcc * (num_context - time_slice)
        # 后序补0
        if time_slice >= need_add_zero_max:
            add_empty_pos = empty_mfcc * (time_slice - need_add_zero_max + 1)
            mfcc_pos = origin_inputs[time_slice + 1:]
        else:
            mfcc_pos = origin_inputs[time_slice + 1:time_slice + num_context + 1]
        mfcc_pre = origin_inputs[max(0, time_slice - num_context):time_slice]
        mfcc_pos = np.reshape(mfcc_pos, [-1])
        mfcc_pre = np.reshape(mfcc_pre, [-1])
        mfcc_con = np.concatenate((add_empty_pre, mfcc_pre, origin_inputs[time_slice], mfcc_pos, add_empty_pos))
        assert len(mfcc_con) == (num_context * 2 + 1) * numcep
        train_inputs[time_slice] = mfcc_con
    train_inputs = (train_inputs - np.mean(train_inputs)) / np.std(train_inputs)  # 标准化
    return train_inputs


# 构建词表
def generate_words_table(labels):
    words = []
    for label in labels:
        words += [word for word in label]
    couter = Counter(words)
    words = sorted(couter)
    print(words)
    length = len(words)
    words_num_map = dict(zip(words, range(length)))
    print(words_num_map)
    return words, words_num_map  # 返回正反向字典


# 将文字转化为字向量
def get_words_vector(label, word_num_map):
    if len(label) == 0:
        return None
    words_size = len(word_num_map)
    transfer_num = lambda word: word_num_map.get(word, words_size)
    tensor = list(map(transfer_num, label))
    return tensor


# 将音频转为mfcc向量，将文字转换为文字向量 并分别输出他们每一个文件的长度
def get_audio_and_transcriptch(audio_files, labels, numcep, n_context, word_num_map):
    input_mfccs = []
    len_input_mfcc = []
    transcriptchs = []
    len_transcriptchs = []
    for audio_file, label in zip(audio_files, labels):
        mfcc = get_audio_mfcc(audio_file, numcep, n_context)
        transcriptch = get_words_vector(label, word_num_map)
        input_mfccs.append(mfcc)
        len_input_mfcc.append(len(mfcc))
        len_transcriptchs.append(len(transcriptch))
        transcriptchs.append(transcriptch)

    return np.array(input_mfccs), \
           np.array(len_input_mfcc), \
           np.array(transcriptchs), \
           np.array(len_transcriptchs)


# 若干个音频的mfcc对齐并补0
def padding_mfcc(input_mfccs, dtype=np.float32, postpadding=True):
    lens = [len(input_mfcc) for input_mfcc in input_mfccs]
    input_len = len(lens)
    max_len = max(lens)
    mfcc_numcep = tuple()
    for input_mfcc in input_mfccs:
        if len(input_mfcc) > 0:
            mfcc_numcep = np.array(input_mfcc).shape[1:]
            break
    padding_inputs = np.zeros((input_len, max_len) + mfcc_numcep, dtype=dtype)  # (8, 584,494)
    for ids, input_mfcc in enumerate(input_mfccs):
        if postpadding:
            padding_inputs[ids, :lens[ids], :] = input_mfcc
        else:
            padding_inputs[ids, -lens[ids]:, :] = input_mfcc
    return padding_inputs, lens


# 将字向量转化为稀疏矩阵存储
def get_sparse_tuple(labels):
    lens = [len(label) for label in labels]
    labels_nun = len(lens)
    indices = []
    values = []
    for idx in range(labels_nun):
        for index, word in enumerate(labels[idx]):
            values.append(word)
            indices.append([idx, index])
    shape = [labels_nun, np.array(indices).max(0)[1] + 1]
    return np.asarray(indices, dtype=np.int32), np.asarray(values, dtype=np.int32), np.asarray(shape, dtype=np.int32)


# 将稀疏矩阵反向转为字向量


def sparse_tuple_to_labels(tuple, words):
    indices = tuple[0]
    values = tuple[1]
    shape = tuple[2]
    result = [''] * shape[0]
    for ids, index in enumerate(indices):
        result[index[0]] += words[values[ids]]
    return result


#  将普通文字向量转化为文字
def tensor_to_labels(tensor, words):
    result = ''
    for index in tensor:
        result += words[index]
    return result


if __name__ == '__main__':
    # np.set_printoptions(threshold=np.inf)
    labels, audios = fi.get_audios_and_labels('/Users/zhouqianyu/resources/data_thchs30/train',
                                              '/Users/zhouqianyu/resources/data_thchs30/data')
    words, word_num_map = generate_words_table(labels)
    input_mfccs, len_input_mfcc, transcriptchs, len_transcriptchs = \
        get_audio_and_transcriptch(audios[:8], labels[:8], 26, 9, word_num_map)
