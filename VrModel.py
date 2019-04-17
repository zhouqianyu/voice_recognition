from deep_learning import *

LEANING_RATE = 0.001  # 初始学习率
LEANING_RATE_DECAY = 0.98  # 学习率衰减

REGULARIZATION_RATE = 0.0001  # 正则化系数
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均模型参数

TRAINING_EPOCH = 100  # 迭代次数(完整过一遍数据)

NUMCEP = 26  # mfcc系数个数
N_CONTEXT = 9  # 上下文个数
BATCH_SIZE = 8

N_HIDDEN_1 = 1024
N_HIDDEN_2 = 1024
N_HIDDEN_3 = 1024 * 2
N_CELL = 1024
N_HIDDEN_5 = 1024

KEEP_DROPOUT_RATE = 0.95
RELU_CLIP = 20  # 避免梯度爆炸进行梯度裁剪

STDDEV = 0.046875


class VrModel:
    def __init__(self, is_training, data_size, vacab_size):
        self.is_training = is_training
        n_mfcc = (2 * N_CONTEXT + 1) * NUMCEP
        self.inputs = tf.placeholder(tf.float32, shape=[None, None, n_mfcc], name='inputs')
        self.targets = tf.sparse_placeholder(tf.int32, name='targets')
        self.seq_length = tf.placeholder(tf.int32, shape=[None], name='seq_length')
        if is_training:
            self.keep_dropout = tf.placeholder(tf.float32)

        self.global_step = tf.Variable(0, trainable=False)

        input_tensors = tf.transpose(self.inputs, [1, 0, 2])  # 将inputs转化为时序优先序列
        input_tensors = tf.reshape(input_tensors, [-1, n_mfcc])
        regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

        moving_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, self.global_step)
        learning_rate = tf.train.exponential_decay(LEANING_RATE, self.global_step,
                                                   data_size / BATCH_SIZE, LEANING_RATE_DECAY, staircase=True)
        # 第一层全连接神经网络
        with tf.variable_scope('fc1'):
            w1 = variable_on_cpu([n_mfcc, N_HIDDEN_1], 'w', tf.random_normal_initializer(stddev=STDDEV))
            b1 = variable_on_cpu([N_HIDDEN_1], 'b', tf.random_normal_initializer(stddev=STDDEV))
            layer1 = tf.minimum(tf.nn.relu(tf.matmul(input_tensors, w1) + b1), RELU_CLIP)
            if is_training:
                layer1 = tf.nn.dropout(layer1, self.keep_dropout)
                tf.add_to_collection('losses', regularizer(w1))

        # 第二层全连接神经网络
        with tf.variable_scope('fc2'):
            w2 = variable_on_cpu([N_HIDDEN_1, N_HIDDEN_2], 'w', tf.random_normal_initializer(stddev=STDDEV))
            b2 = variable_on_cpu([N_HIDDEN_2], 'b', tf.random_normal_initializer(stddev=STDDEV))
            layer2 = tf.minimum(tf.nn.relu(tf.matmul(layer1, w2) + b2), RELU_CLIP)
            if is_training:
                layer2 = tf.nn.dropout(layer2, self.keep_dropout)
                tf.add_to_collection('losses', regularizer(w2))

        # 第三层全连接神经网路
        with tf.variable_scope('fc3'):
            w3 = variable_on_cpu([N_HIDDEN_2, N_HIDDEN_3], 'w', tf.random_normal_initializer(stddev=STDDEV))
            b3 = variable_on_cpu([N_HIDDEN_3], 'b', tf.random_normal_initializer(stddev=STDDEV))
            layer3 = tf.minimum(tf.nn.relu(tf.matmul(layer2, w3) + b3), RELU_CLIP)
            if is_training:
                layer3 = tf.nn.dropout(layer3, self.keep_dropout)
                tf.add_to_collection('losses', regularizer(w3))

        layer3 = tf.reshape(layer3, [-1, BATCH_SIZE, N_HIDDEN_3])  # 换成三维的时序优先进入循环神经网络层
        # 双向循环神经网络层
        with tf.variable_scope('bi-rnn'):
            lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(N_CELL)
            lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_fw, input_keep_prob=self.keep_dropout)
            lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(N_CELL)
            lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_bw, input_keep_prob=self.keep_dropout)

            outputs, self.state = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cell_fw,
                                                                  cell_bw=lstm_cell_bw, inputs=layer3,
                                                                  sequence_length=self.seq_length, time_major=True,
                                                                  dtype=tf.float32)
            outputs = tf.concat(outputs, 2)  # 连接两个神经元输出结果
            layer4 = tf.reshape(outputs, [-1, 2 * N_CELL])
        # 第五层全连接神经网络
        with tf.variable_scope('fc5'):
            w5 = variable_on_cpu([2 * N_CELL, N_HIDDEN_5], 'w', tf.random_normal_initializer(stddev=STDDEV))
            b5 = variable_on_cpu([N_HIDDEN_5], 'b', tf.random_normal_initializer(stddev=STDDEV))
            layer5 = tf.minimum(tf.nn.relu(tf.matmul(layer4, w5) + b5), RELU_CLIP)
            if is_training:
                tf.nn.dropout(layer5, self.keep_dropout)
                tf.add_to_collection('losses', regularizer(w5))

        # 连向输出层
        with tf.variable_scope('fc6'):
            w6 = variable_on_cpu([N_HIDDEN_5, vacab_size], 'w', tf.random_normal_initializer(stddev=STDDEV))
            b6 = variable_on_cpu([vacab_size], 'b', tf.random_normal_initializer(stddev=STDDEV))
            layer6 = tf.matmul(layer5, w6) + b6
            if is_training:
                tf.add_to_collection('losses', regularizer(w6))

        logits = tf.reshape(layer6, [-1, BATCH_SIZE, vacab_size])  # 转化成时序优先输出
        if is_training:
            moving_average_op = moving_average.apply(tf.trainable_variables())
            self.avg_loss = tf.reduce_mean(tf.nn.ctc_loss(self.targets, logits, self.seq_length))  # 利用ctc计算loss
            loss = self.avg_loss
            train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=self.global_step)

            with tf.control_dependencies([moving_average_op, train]):
                self.train_op = tf.no_op('train')

        # 使用ctc_decoder进行解码

        self.decode, log_prob = tf.nn.ctc_beam_search_decoder(logits, self.seq_length, merge_repeated=False)
        # 计算与正确结果之间的编辑距离
        self.distance = tf.reduce_mean(tf.edit_distance(tf.cast(self.decode[0], tf.int32), self.targets)) #开始速度超级慢

        # 其中decode[0] 为一个稀疏矩阵

    def run(self, sess, dict_map, eval=False):
        if self.is_training:
            _, avg_loss, global_step = sess.run([self.train_op,
                                                 self.avg_loss, self.global_step], feed_dict=dict_map)
            return avg_loss, global_step
        else:
            if eval:
                result, distance = sess.run([self.decode[0], self.distance], feed_dict=dict_map)
                return result, distance
            else:
                result = sess.run(self.decode[0], feed_dict=dict_map)
                return result

if __name__ == '__main__':
    print(tf.__version__)
