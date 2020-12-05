import os
import torch

from tokenizers import BertWordPieceTokenizer
from datasets.mrc_ner_dataset import MRCNERDataset
from datasets.collate_functions import collate_to_max_length
from torch.utils.data import DataLoader
from finetuning_argparse import get_argparse
from models.bert_query_ner import BertQueryNER
from models.query_ner_config import BertQueryNerConfig
from torch.nn.modules import BCEWithLogitsLoss
from utils import category
from metrics.functional.query_span_f1 import query_span_f1_all_tag


def compute_loss(start_logits, end_logits, span_logits,
                 start_labels, end_labels, match_labels, start_label_mask, end_label_mask):
    bce_loss = BCEWithLogitsLoss(reduction="none")

    batch_size, seq_len = start_logits.size()

    start_float_label_mask = start_label_mask.view(-1).float()
    end_float_label_mask = end_label_mask.view(-1).float()
    match_label_row_mask = start_label_mask.bool().unsqueeze(-1).expand(-1, -1, seq_len)
    match_label_col_mask = end_label_mask.bool().unsqueeze(-2).expand(-1, seq_len, -1)
    match_label_mask = match_label_row_mask & match_label_col_mask
    match_label_mask = torch.triu(match_label_mask, 0)  # start should be less equal to end

    float_match_label_mask = match_label_mask.view(batch_size, -1).float()

    start_loss = bce_loss(start_logits.view(-1), start_labels.view(-1).float())
    start_loss = (start_loss * start_float_label_mask).sum() / start_float_label_mask.sum()
    end_loss = bce_loss(end_logits.view(-1), end_labels.view(-1).float())
    end_loss = (end_loss * end_float_label_mask).sum() / end_float_label_mask.sum()
    match_loss = bce_loss(span_logits.view(batch_size, -1), match_labels.view(batch_size, -1).float())
    match_loss = match_loss * float_match_label_mask
    match_loss = match_loss.sum() / (float_match_label_mask.sum() + 1e-10)

    return start_loss, end_loss, match_loss


def dev(model, dataloader, args):
    output_tp, output_fp, output_fn = [], [], []
    for step, batch in enumerate(dataloader):
        model.eval()
        print(step)
        tokens, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels, sample_idx, label_idx = batch
        attention_mask = (tokens != 0).long()
        tokens, attention_mask, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels \
            = tokens.to(device), attention_mask.to(device), token_type_ids.to(device), start_labels.to(
            device), end_labels.to(device), start_label_mask.to(device), end_label_mask.to(device), match_labels.to(
            device)
        with torch.no_grad():
            start_logits, end_logits, span_logits = model(input_ids=tokens, token_type_ids=token_type_ids,
                                                          attention_mask=attention_mask)

            start_loss, end_loss, match_loss = compute_loss(start_logits=start_logits,
                                                            end_logits=end_logits,
                                                            span_logits=span_logits,
                                                            start_labels=start_labels,
                                                            end_labels=end_labels,
                                                            match_labels=match_labels,
                                                            start_label_mask=start_label_mask,
                                                            end_label_mask=end_label_mask
                                                            )

        total_loss = args.weight_start * start_loss + args.weight_end * end_loss + args.weight_span * match_loss
        start_preds, end_preds = start_logits > 0, end_logits > 0
        tp, fp, fn, match_preds = query_span_f1_all_tag(start_preds=start_preds, end_preds=end_preds,
                                                        match_logits=span_logits,
                                                        start_label_mask=start_label_mask,
                                                        end_label_mask=end_label_mask,
                                                        match_labels=match_labels)

        # print('dev_span_f1: ', span_f1_stats)
        output_tp = output_tp + tp.tolist()
        output_fp = output_fp + fp.tolist()
        output_fn = output_fn + fn.tolist()

    tag_tp = [0] * len(category)
    tag_fp = [0] * len(category)
    tag_fn = [0] * len(category)

    assert len(output_tp) == len(output_fp) == len(output_fn)

    length = len(output_tp)
    print('长度为：', length)
    for i in range(length):
        tag_tp[i % len(category)] += output_tp[i]
        tag_fp[i % len(category)] += output_fp[i]
        tag_fn[i % len(category)] += output_fn[i]

    for i in range(len(category)):
        recall = tag_tp[i] / (tag_tp[i] + tag_fn[i] + 1e-10)
        precision = tag_tp[i] / (tag_tp[i] + tag_fp[i] + 1e-10)
        f1_score = recall * precision * 2 / (recall + precision + 1e-10)
        print('当前类别为{0}，recall:{1},precision:{2},f1-score:{3}'.format(category[i], recall, precision, f1_score))

    recall = sum(output_tp) / (sum(output_tp) + sum(output_fn) + 1e-10)
    precision = sum(output_tp) / (sum(output_tp) + sum(output_fp) + 1e-10)
    f1_score = recall * precision * 2 / (recall + precision + 1e-10)

    print('总的recall:{0},precision:{1},f1-score:{2}'.format(recall, precision, f1_score))
    return recall, precision, f1_score


if __name__ == '__main__':
    args = get_argparse().parse_args()
    weight_sum = args.weight_start + args.weight_end + args.weight_span
    args.weight_start = args.weight_start / weight_sum
    args.weight_end = args.weight_end / weight_sum
    args.weight_span = args.weight_span / weight_sum

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

    train_json_path = os.path.join(json_path, 'mrc-ner.train')
    dev_json_path = os.path.join(json_path, 'mrc-ner.dev')
    test_json_path = os.path.join(json_path, 'mrc-ner.test')

    dev_dataset = MRCNERDataset(json_path=dev_json_path, tokenizer=tokenizer, possible_only=False,
                                is_chinese=is_chinese)
    dev_dataloader = DataLoader(dev_dataset, batch_size=1,
                                collate_fn=collate_to_max_length)

    print('----------------------------------------------------------------------')

    span_recall, span_precision, span_f1 = dev(model, dev_dataloader, args)
    print('recall:{:f} ,precision:{:f} ,f1_score:{:f}'.format(span_recall, span_precision, span_f1))
