# -*- coding: utf-8 -*-
# Some functions come from the Internet, if you violate your rights, please contact us.
import os
from itertools import chain

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

SPECIAL_TOKENS = ["[CLS]", "[SEP]", "[speaker1]", "[speaker2]"]
MODEL_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]
KNOWLEDGE_SPECIAL_TOKENS = ['[text]', '[/text]', '[box]', '[/box]', '[graph]', '[/graph]',
                            '[TEXT_SEP]', '[KV_SEP]', '[HR_SEP]', '[TR_SEP]',
                            '[field]', '[/field]', '[triple]', '[/triple]', '[none_knowledge]',
                            '[text_knowledge_mark]', '[infobox_knowledge_mark]', '[graph_knowledge_mark]']


class WBDataset(Dataset):

    def __init__(self, args, data_src, data_tgt, knowledge, tokenizer, max_history=15, batch_first=True, lm_labels=True):
        self.data_src = data_src
        self.data_tgt = data_tgt
        self.knowledge = knowledge
        self.tokenizer = tokenizer
        self.max_history = max_history
        self.pad = tokenizer.pad_token_id
        self.batch_first = batch_first
        self.lm_labels = lm_labels
        self.args = args

    def __len__(self):
        return len(self.data_src)

    def __getitem__(self, index):
        if self.lm_labels:
            history = self.data_src[index]
            response = self.data_tgt[index]
        else:
            history = self.data_src[index]
            response = []
        knowledge = {}
        for key, value in self.knowledge.items():
            knowledge[key] = value[index]
        return self.process(history, response, knowledge, self.args.knowledge_mask)

    def process(self, history, response, knowledge, knowledge_mask="", max_len=256):
        bos, eos, speaker1, speaker2 = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
        text_b, text_e, box_b, box_e, graph_b, graph_e, \
        TEXT_SEP, KV_SEP, HR_SEP, TR_SEP, \
        field_b, field_e, triple_b, triple_e, none_knowledge, \
        text_knowledge, infobox_knowledge, graph_knowledge = self.tokenizer.convert_tokens_to_ids(KNOWLEDGE_SPECIAL_TOKENS)


        # mask_flag
        mask_flag = set()
        if knowledge_mask.find('graph') > -1:
            mask_flag.add('graph')
        if knowledge_mask.find('infobox') > -1:
            mask_flag.add('infobox')
        if knowledge_mask.find('text') > -1:
            mask_flag.add('text')

        # Estimate the data len
        # Text
        text_len = 2
        if 'text' not in knowledge or knowledge['text'] == [] or 'text' in mask_flag:
            text_len += 1
        else:
            for seq in knowledge['text']:
                text_len += len(seq) + 1
                # text.extend(seq + [TEXT_SEP])
            # the last token is seq, should remove it
            text_len -= 1

        # Infobox
        infobox_len = 2
        if 'infobox' not in knowledge or knowledge['infobox'] == [] or 'infobox' in mask_flag:
            infobox_len += 1
        else:
            for i in knowledge['infobox']:
                infobox_len += 1 + len(i[0]) + 1 + len(i[1]) + 1
                # infobox.extend([field_b] + i[0] + [KV_SEP] + i[1] + [field_e])

        # Graph
        graph_len = 2
        if 'graph' not in knowledge or knowledge['graph'] == [] or 'graph' in mask_flag:
            graph_len += 1
        else:
            for i in knowledge['graph']:
                graph_len += 1 + len(i[0]) + 1 + len(i[1]) + 1 + len(i[2]) + 1
                # graph.extend([triple_b] + i[0] + [HR_SEP] + i[1] + [TR_SEP] + i[2] + [triple_e])

        # Total Len
        total_len = 1 + text_len + infobox_len + graph_len + 1 + 1 + len(history) + 1 + len(response) + 1
        # sequence = [bos] + text + infobox + graph + [eos] + [speaker1] + history + [speaker2] + response + [eos]

        if total_len > max_len:
            # to clip: [bos] [eos] + [speaker1] + history + [speaker2] + response + [eos]
            valid_knowledge_length = max_len - (1 + 1 + 1 + len(history) + 1 + len(response) + 1)
            # ensure the min length
            valid_knowledge_length -= 9
            current_knowledge_length = text_len + infobox_len + graph_len
            cut_off_ratio = valid_knowledge_length / current_knowledge_length
            max_text_len = int(text_len * cut_off_ratio) + 3
            max_infobox_len = int(infobox_len * cut_off_ratio) + 3
            max_graph_len = int(graph_len * cut_off_ratio) + 3
        else:
            max_text_len = max_len
            max_infobox_len = max_len
            max_graph_len = max_len


        # process text knowledge
        text = [text_b]
        if 'text' not in knowledge or knowledge['text'] == [] or 'text' in mask_flag:
            text.append(none_knowledge)
        else:
            for seq in knowledge['text']:
                text.extend(seq + [TEXT_SEP])
            # the last token is seq, should remove it
            text = text[:-1]
        text = text[0:max_text_len - 1]
        text.append(text_e)
        assert len(text) <= max_text_len, len(text)

        # process infobox knowledge
        infobox = [box_b]
        if 'infobox' not in knowledge or knowledge['infobox'] == [] or 'infobox' in mask_flag:
            infobox.append(none_knowledge)
        else:
            for i in knowledge['infobox']:
                if len(infobox) + 1 + (1 + len(i[0]) + 1 + len(i[1]) + 1) <= max_infobox_len:
                    infobox.extend([field_b] + i[0] + [KV_SEP] + i[1] + [field_e])
                else:
                    break
        infobox.append(box_e)
        assert len(infobox) <= max_infobox_len, len(infobox)

        # process graph knowledge
        graph = [graph_b]
        if 'graph' not in knowledge or knowledge['graph'] == [] or 'graph' in mask_flag:
            graph.append(none_knowledge)
        else:
            for i in knowledge['graph']:
                if len(graph) + 1 + (1 + len(i[0]) + 1 + len(i[1]) + 1 + len(i[2]) + 1) <= max_graph_len:
                    graph.extend([triple_b] + i[0] + [HR_SEP] + i[1] + [TR_SEP] + i[2] + [triple_e])
                else:
                    break
        graph.append(graph_e)
        assert len(graph) <= max_graph_len, len(graph)

        sequence = [bos] + text + infobox + graph + [eos] + [speaker1] + history + [speaker2] + response + [eos]
        assert len(sequence) <= max_len, len(sequence)
        instance = {}
        instance["input_ids"] = sequence
        instance["token_type_ids"] = [bos] + [text_knowledge] * len(text) + [infobox_knowledge] * len(infobox) + \
                                     [graph_knowledge] * (len(graph) + 1) + \
                                     [speaker1] * (len(history) + 1) + [speaker2] * (len(response) + 2)
        assert len(instance["input_ids"]) == len(instance["token_type_ids"])
        instance["lm_labels"] = [-1] * len(instance["input_ids"])
        if self.lm_labels:
            instance["lm_labels"][-(len(response) + 1):] = response + [eos]

        return instance

    def collate(self, batch):
        input_ids = pad_sequence(
            [torch.tensor(instance["input_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        token_type_ids = pad_sequence(
            [torch.tensor(instance["token_type_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        labels = pad_sequence(
            [torch.tensor(instance["lm_labels"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=-1)
        return input_ids, token_type_ids, labels


class DatasetBase(Dataset):

    def __init__(self, data_path=None):
        self.data_path = data_path
        self.data_files = list()
        self.data_files_offset = list()
        self.data_len = 0
        self._check_files()

    def _check_files(self):
        if self.data_path is None:
            raise RuntimeError("Data path cannot be \
                empty at same time.")

        if self.data_path:
            if not os.path.exists(self.data_path):
                raise RuntimeError("Training files does not exist at " + self.data_path)
            prepare_files_offset(self.data_path, self.data_files,
                                 self.data_files_offset)
            self.data_len = len(self.data_files_offset)

    def __len__(self):
        return self.data_len

    def _get_line(self, index):
        tup = self.data_files_offset[index]
        target_file = self.data_files[tup[0]]
        with open(target_file, "r", encoding="utf-8") as f:
            f.seek(tup[1])
            line = f.readline()
        return line


class WBdistDataset(DatasetBase):

    def __init__(self, tokenizer, max_history=15, batch_first=True, lm_labels=True, *inputs, **kwargs):
        super(WBdistDataset, self).__init__(*inputs, **kwargs)
        self.tokenizer = tokenizer
        self.max_history = max_history
        self.pad = tokenizer.pad_token_id
        self.batch_first = batch_first
        self.lm_labels = lm_labels

    def __getitem__(self, index):
        tokenizer = self.tokenizer
        dialog = self._get_line(index)
        dialog = dialog.strip().split("\t")

        def tokenize(obj):
            if isinstance(obj, str):
                return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) for n, o in obj.items())
            return list(tokenize(o) for o in obj)

        dialog = tokenize(dialog)
        history = dialog[:-1]
        candidates = dialog[-1]
        return self.process(history, candidates)

    def process(self, history, response, with_eos=True):
        bos, eos, speaker1, speaker2 = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
        sequence = [[bos]] + history + [response + ([eos] if with_eos else [])]
        sequence = [sequence[0]] + [[speaker2 if i % 2 else speaker1] + s
                                    for i, s in enumerate(sequence[1:])]
        instance = {}
        instance["input_ids"] = list(chain(*sequence))
        instance["token_type_ids"] = [bos] + [speaker2 if i % 2 else speaker1 for i, s in
                                              enumerate(sequence[1:])
                                              for _ in s]
        instance["lm_labels"] = [-1] * len(instance["input_ids"])
        if self.lm_labels:
            instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:]

        return instance

    def collate(self, batch):
        input_ids = pad_sequence(
            [torch.tensor(instance["input_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        token_type_ids = pad_sequence(
            [torch.tensor(instance["token_type_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        labels = pad_sequence(
            [torch.tensor(instance["lm_labels"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=-1)
        return input_ids, token_type_ids, labels


def prepare_files_offset(path, files_list, offset_list):
    """Fill the file index and offsets of each line in files_list in offset_list
    Args:
        path: string of file path, support single file or file dir
        files_list: the list contains file names
        offset_list: the list contains the tuple of file name index and offset
    """
    if os.path.isdir(path):  # for multi-file, its input is a dir
        files_list.extend([os.path.join(path, f) for f in os.listdir(path)])
    elif os.path.isfile(path):  # for single file, its input is a file
        files_list.append(path)
    else:
        raise RuntimeError(path + " is not a normal file.")
    for i, f in enumerate(files_list):
        offset = 0
        with open(f, "r", encoding="utf-8") as single_file:
            for line in single_file:
                tup = (i, offset)
                offset_list.append(tup)
                offset += len(bytes(line, encoding='utf-8'))
