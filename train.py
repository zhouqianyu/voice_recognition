from file_import import *
from VrModel import *
AUDIO_PATH = '/Users/zhouqianyu/resources/data_thchs30/train'
LABEL_PATH = '/Users/zhouqianyu/resources/data_thchs30/data'
Test_AUDIO = '/Users/zhouqianyu/resources/data_thchs30/test'
model_save_path = 'saver/'
model_name = 'vr_model.ckpt'

def main(argv=None):
    labels, audios = get_audios_and_labels(AUDIO_PATH, LABEL_PATH)
    words, word_num_map = generate_words_table(labels)
    train_model = VrModel(True, len(audios), len(words) + 1)  # +1是给ctc的blank label
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(model_save_path)
    with tf.Session() as sess:
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            tf.global_variables_initializer().run()
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("logs/", sess.graph)
        for train_step in range(TRAINING_EPOCH):
            train_cost = 0
            itr = iter(next_batch(BATCH_SIZE, audios, labels, NUMCEP, N_CONTEXT, word_num_map))
            for batch_step, (input_labels, lens, targets) in enumerate(itr):
                dict_map = {train_model.inputs: input_labels,
                            train_model.targets: targets,
                            train_model.seq_length: lens,
                            train_model.keep_dropout: KEEP_DROPOUT_RATE}
                avg_loss, global_step, rs, = train_model.run(sess, dict_map, merged)
                train_cost += avg_loss
                writer.add_summary(rs, global_step)
                if batch_step % 1 == 0:
                    print('目前正在进行第%d轮，第%d次迭代，总轮数%d, 当前损失率为%f' % (train_step + 1,
                                                          batch_step + 1, global_step, train_cost / (batch_step + 1)))

                    with open('./log.txt', 'a') as f:
                        print('目前正在进行第%d轮，第%d次迭代，总轮数%d, 当前损失率为%f' % (train_step + 1,
                                                                     batch_step + 1, global_step,
                                                                     train_cost / (batch_step + 1)), file=f, flush=True)
                if global_step % 100 == 0:
                    saver.save(sess, os.path.join(model_save_path, model_name), global_step)


if __name__ == '__main__':
    tf.app.run()
