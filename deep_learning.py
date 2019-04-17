import tensorflow as tf
from mfcc_and_labels import *
from math import *

#  迭代器迭代生成一个batch_size的训练参数


def next_batch(batch_size, audio_files, all_labels, numcep, n_context, word_num_map):
    current_id = 0
    assert len(audio_files) == len(all_labels)
    data_num = len(audio_files)
    epoch = ceil(data_num / batch_size)
    for i in range(epoch):
        audios = audio_files[current_id:min(data_num, current_id + batch_size)]
        labels = all_labels[current_id:min(data_num, current_id + batch_size)]
        current_id += batch_size
        inputs, len_inputs, targets, len_tatgets = \
            get_audio_and_transcriptch(audios, labels,
                                           numcep, n_context,
                                           word_num_map)
        inputs, len_input = padding_mfcc(inputs)
        sparse_target = get_sparse_tuple(targets)
        yield inputs, len_input, sparse_target


def variable_on_cpu(shape, name, initializer, dtype=tf.float32):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name=name, shape=shape, dtype=dtype, initializer=initializer)
    return var


if __name__ == '__main__':
    labels, audios = fi.get_audios_and_labels('/Users/zhouqianyu/resources/data_thchs30/train',
                                              '/Users/zhouqianyu/resources/data_thchs30/data')
    words, word_num_map = generate_words_table(labels)
    itr = iter(next_batch(8, audios[:99], labels[:99], 26, 9, word_num_map))
    for idx, (a, b, c) in enumerate(itr):
        print(labels[idx*8])
        print(sparse_tuple_to_labels(c, words))
