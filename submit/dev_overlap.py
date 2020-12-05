import pandas as pd


def dev_label_chong(train_label):
    count = set()
    pre_ID = 0
    ID_union = set()
    for row_index, row in train_label.iterrows():
        assert len(row) == 5
        ID, Category, Pos_b, Pos_e, Privacy = row['ID'], row['Category'], row['Pos_b'], row['Pos_e'], row[
            'Privacy']
        start, end = int(Pos_b), int(Pos_e)

        now_set = set()
        begin = start
        while begin <= end:
            now_set.add(begin)
            begin += 1
        if ID == pre_ID:
            if len(ID_union.intersection(now_set)) > 0:
                count.add(ID)
        else:
            pre_ID = ID
            ID_union = set()
        ID_union = ID_union.union(now_set)
    return count


test_label_path = './predict_position.csv'
test_label = pd.read_csv(test_label_path)
count = dev_label_chong(test_label)
count = list(count)
print(len(count))
print('test_label:', count)
