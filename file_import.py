import os


# 该文件主要是读取音频文件以及其对应的翻译文字


# 取出音频文件
def get_audio_files(audios_path):
    audio_files = []
    results = os.walk(audios_path)
    for (path, dirname, files) in results:
        for file in files:
            if file.endswith('wav') or file.endswith('Wav'):
                audio_files.append(os.path.join(path, file))
    return audio_files


# 根据音频文件取出对应的文字
def get_labels(audio_files, trn_path):
    labels = []
    for audio_file in audio_files:
        path, file = os.path.split(audio_file)
        label_path = os.path.join(trn_path, file + '.trn')
        if os.path.exists(label_path):
            label_file = open(label_path, 'r')
        else:
            continue
        label = label_file.readline()
        labels.append(label.split('\n')[0])
    return labels


# 综合以上两个方法 获取音频与对应文字
def get_audios_and_labels(audios_path, lables_path):
    audios = get_audio_files(audios_path)
    labels = get_labels(audios, lables_path)
    return labels, audios




if __name__ == '__main__':
    labels, audios = get_audios_and_labels('/Users/zhouqianyu/resources/data_thchs30/train',
                                           '/Users/zhouqianyu/resources/data_thchs30/data')
    print(labels[1], audios[1])
    print(len(labels), len(audios))
