import os
import torch
import json

from tokenizers import BertWordPieceTokenizer
from datasets.mrc_ner_dataset import MRCNERDataset_test
from torch.utils.data import DataLoader
from finetuning_argparse import get_argparse
from models.bert_query_ner import BertQueryNER
from models.query_ner_config import BertQueryNerConfig
from utils import load_pickle, category


def extract_flat_spans(start_pred, end_pred, match_pred, label_mask, id, tag, now_tokens_str):
    """
    Extract flat-ner spans from start/end/match logits
    Args:
        start_pred: [seq_len], 1/True for start, 0/False for non-start
        end_pred: [seq_len, 2], 1/True for end, 0/False for non-end
        match_pred: [seq_len, seq_len], 1/True for match, 0/False for non-match
        label_mask: [seq_len], 1 for valid boundary.
    Returns:
        tags: list of tuple (start, end)
    Examples:
        >>> start_pred = [0, 1]
        >>> end_pred = [0, 1]
        >>> match_pred = [[0, 0], [0, 1]]
        >>> label_mask = [1, 1]
        >>> extract_flat_spans(start_pred, end_pred, match_pred, label_mask)
        [(1, 2)]
    """
    json_d = {}
    entities = []

    bmes_labels = ["O"] * len(start_pred)
    start_positions = [idx for idx, tmp in enumerate(start_pred) if tmp and label_mask[idx]]
    end_positions = [idx for idx, tmp in enumerate(end_pred) if tmp and label_mask[idx]]

    for start_item in start_positions:
        bmes_labels[start_item] = f"B-{tag}"
    for end_item in end_positions:
        bmes_labels[end_item] = f"E-{tag}"

    for tmp_start in start_positions:
        tmp_end = [tmp for tmp in end_positions if tmp >= tmp_start]
        if len(tmp_end) == 0:
            continue
        else:
            tmp_end = min(tmp_end)
        if match_pred[tmp_start][tmp_end]:
            if tmp_start != tmp_end:
                for i in range(tmp_start + 1, tmp_end):
                    bmes_labels[i] = f"M-{tag}"
                # entities.append([tag, tmp_start - 11, tmp_end- 11, "".join(now_tokens_str[tmp_start:tmp_end + 1])])
                entities.append([tag, tmp_start - 11, tmp_end - 11])
            else:
                bmes_labels[tmp_end] = f"S-{tag}"
                # entities.append([tag, tmp_start - 11, tmp_end - 11, "".join(now_tokens_str[tmp_start:tmp_end + 1])])
                entities.append([tag, tmp_start - 11, tmp_end - 11])
    json_d['id'] = id
    json_d['entities'] = entities
    return json_d


if __name__ == '__main__':
    args = get_argparse().parse_args()
    bert_path = args.bert_config_dir
    json_path = args.data_dir
    is_chinese = True
    vocab_file = os.path.join(bert_path, "vocab.txt")
    tokenizer = BertWordPieceTokenizer(vocab_file=vocab_file)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    output_dir = os.path.join(args.output_dir, "best_f1_checkpoint")

    bert_config = BertQueryNerConfig.from_pretrained(output_dir,
                                                     hidden_dropout_prob=args.bert_dropout,
                                                     attention_probs_dropout_prob=args.bert_dropout,
                                                     mrc_dropout=args.mrc_dropout)
    model = BertQueryNER.from_pretrained(output_dir, config=bert_config).to(device)
    model.eval()

    test_json_path = os.path.join(json_path, 'mrc-ner.test')
    test_dataset = MRCNERDataset_test(json_path=test_json_path, tokenizer=tokenizer,
                                      is_chinese=is_chinese)
    test_dataloader = DataLoader(test_dataset, batch_size=1, )

    all_test_data = json.load(open(test_json_path, encoding="utf-8"))

    print(len(all_test_data))

    results = []

    sentences_seg_path = os.path.join(args.data_dir, 'test_sentences_seg.pkl')
    all_sentences_seg = load_pickle(sentences_seg_path)
    id = -1

    print(len(test_dataloader))

    for index, batch in enumerate(test_dataloader):

        print(index)

        now_test_data = all_test_data[index]

        tokens, token_type_ids, label_mask, sample_idx, label_idx = batch

        attention_mask = (tokens != 0).long()

        tokens, attention_mask, token_type_ids, label_mask = tokens.to(device), attention_mask.to(
            device), token_type_ids.to(device), label_mask.to(device)

        start_logits, end_logits, span_logits = model(input_ids=tokens, token_type_ids=token_type_ids,
                                                      attention_mask=attention_mask)

        start_logits, end_logits, span_logits = (start_logits > 0).long(), (end_logits > 0).long(), (
                span_logits > 0).long()

        start_logits, end_logits, span_logits, label_mask = start_logits.squeeze(dim=0).tolist(), \
                                                            end_logits.squeeze(dim=0).tolist(), \
                                                            span_logits.squeeze(dim=0).tolist(), \
                                                            label_mask.squeeze(dim=0).tolist()
        sample_idx = sample_idx.item()
        label_idx = label_idx.item()

        seq_len = len(start_logits)
        now_tokens_str = ['[CLS]'] + list(now_test_data['query']) + ['[SEP]'] + list(now_test_data['context']) + [
            '[SEP]']
        now_tokens_str = now_tokens_str[:seq_len]
        now_tokens_str[-1] = ['[SEP]']

        assert len(now_tokens_str) == len(start_logits)

        if all_sentences_seg[sample_idx][0] == 1 and label_idx == 0:
            id += 1  # 计数器

        json_d = extract_flat_spans(start_logits, end_logits, span_logits, label_mask, id, category[label_idx],
                                    now_tokens_str)
        results.append(json_d)

    for tag in range(len(category)):
        single_results = []
        for index, result in enumerate(results):
            if index % len(category) == tag:
                single_results.append(result)
        output_predict_file = os.path.join(args.output_dir,
                                           'output_predict/test_predict_mrc_{0}.json'.format(category[tag]))
        with open(output_predict_file, "w", encoding='utf-8') as writer:
            for record in single_results:
                writer.write(json.dumps(record, ensure_ascii=False) + '\n')
