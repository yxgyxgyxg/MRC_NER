import os
import json
import pandas as pd

from utils import load_json, load_pickle,category

label_list = ["address", "book", "company", 'game', 'government', 'movie', 'name',
              'organization', 'position', 'scene', 'mobile', 'email', 'QQ', 'vx', ]


def write_to_result(test_data_dir, test_data_json_dir, test_prediction_path, submit_path, all_sentences_seg):
    test_prediction = load_json(test_prediction_path)
    test_text = []
    with open(os.path.join(test_data_json_dir), 'r', encoding='utf-8') as fr:
        for line in fr:
            test_text.append(json.loads(line))
    all_privacy = []

    print(len(all_sentences_seg))
    print(len(test_prediction))
    print(len(test_text))

    for x, y, z in zip(test_text, test_prediction, all_sentences_seg):
        ID = x['id']
        words = list(x['text'])
        entities = y['entities']

        assert x['id'] == y['id']

        test_data_path = test_data_dir + str(ID) + '.txt'
        with open(test_data_path, 'r', encoding='utf-8') as f:
            lines_text = f.readlines()
            raw_text = ''
            for line_text in lines_text:
                raw_text += line_text
        print(ID)
        if len(entities) != 0:
            for subject in entities:
                tag = subject[0]
                start = subject[1]
                end = subject[2]
                word = "".join(words[start:end + 1])

                if z[0] > 1:  # 长度需要添加回去
                    start, end = z[1] + start, z[1] + end
                assert end < len(raw_text)
                assert word == raw_text[start:end + 1]  # 用于验证数据出没出错

                assert tag in label_list

                if '\n' in word:
                    if word[0] == '\n':
                        start, end = start - 1, end - 1
                    else:
                        end = end - 1
                    word = word.replace('\n', '')

                    print('含有换行符的索引：', ID)
                all_privacy.append([ID, tag, start, end, word])
    csv_head = ['ID', 'Category', 'Pos_b', 'Pos_e', 'Privacy']
    df1 = pd.DataFrame(all_privacy, columns=csv_head)
    df1.to_csv(submit_path, encoding='utf-8', index=None)
    return all_privacy


test_data_dir = '../data/ccfner/test/'
test_data_json_dir = '../data/ccfner/test.json'
sentences_seg_path = '../data/ccfner/test_sentences_seg.pkl'

all_privacy = []
for tag in category:
    test_prediction_path = '../output/output_predict/test_predict_mrc_{0}.json'.format(tag)
    submit_path = 'predict/predict_position_{0}.csv'.format(tag)
    all_sentences_seg = load_pickle(sentences_seg_path)
    tag_all_privacy = write_to_result(test_data_dir, test_data_json_dir, test_prediction_path, submit_path,
                                      all_sentences_seg)
    all_privacy = all_privacy + tag_all_privacy

submit_path = 'predict_position.csv'
csv_head = ['ID', 'Category', 'Pos_b', 'Pos_e', 'Privacy']
all_privacy_df = pd.DataFrame(all_privacy, columns=csv_head)
all_privacy_df.to_csv(submit_path, encoding='utf-8', index=None)
