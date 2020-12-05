Implementation of 《a unified MRC framework for named entity recognition》ACL 2020

pytorch:1.5.1

如何运行？
1.准备数据与bert模型
        1.1 data/ccfner2mrc.py  为每条样本构造query
        1.2 bert中文训练模型放到prev_trained_model/bert-base 之下

2.训练 train.py
        2.1 加载数据，datasets/mrc_ner_dataset.py形成datasets，包括tokens、type_id、start_labels、
                end_labels、start_label_mask、end_label_mask、match_label。
        2.2 将数据喂给模型 模型主要包括三方面的损失：start_label损失、end_label损失和match_label损失

3.验证 dev.py
        3.1 加载验证数据，喂给模型，得到三个logits【start_logits, end_logits, span_logits】
        3.2 利用三个logits计算label是否match，主要用到metrics/functional/query_span_f1.py

4.预测 test.py
        4.1 加载测试数据，喂给模型，得到三个logits【start_logits, end_logits, span_logits】
        4.2 执行extract_flat_spans()函数，提取出实体
        


