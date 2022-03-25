# -*- coding: utf-8 -*-
import os
import json

import torch
from torch.utils.data import DataLoader
from transformers import cached_path

from od.inputters.dataset_wb import WBDataset, WBdistDataset

LCCC_URL = "https://coai-dataset.oss-cn-beijing.aliyuncs.com/CleanWB.zip"
SPECIAL_TOKENS = ["[CLS]", "[SEP]", "[speaker1]", "[speaker2]"]


def get_data(args, tokenizer, dataset_path, dataset_cache, knowledge_cache, logger, knowledge_type):
    """ Get tokenized dataset from COTK or cache."""
    assert dataset_path != ""
    dataset_path = dataset_path or LCCC_URL
    dataset_cache = dataset_cache + '_' + type(tokenizer).__name__
    knowledge_cache = knowledge_cache + '_' + type(tokenizer).__name__
    if dataset_cache and os.path.isfile(dataset_cache + '.src'):
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        dataset_src = torch.load(dataset_cache + '.src')
        dataset_tgt = torch.load(dataset_cache + '.tgt')
        print('src', len(dataset_src))
        print('tgt', len(dataset_tgt))
    else:
        # encode dataset
        logger.info("Download dataset from %s", dataset_path)
        cache_file = cached_path(dataset_path)
        with open(cache_file, "r", encoding="utf-8") as f:
            datasets = json.loads(f.read())
        dataset_src = generate_context_dataset(tokenizer, datasets, dataset_cache, logger, 'src')
        dataset_tgt = generate_context_dataset(tokenizer, datasets, dataset_cache, logger, 'tgt')
        print('src', len(dataset_src))
        print('tgt', len(dataset_tgt))

    if knowledge_cache and os.path.isfile(knowledge_cache):
        logger.info("Load tokenized dataset from cache at %s", knowledge_cache)
        knowledge_set = torch.load(knowledge_cache)
        print('knowledge_set', len(knowledge_set))
    else:
        # encode knowledge set
        cache_file = cached_path(dataset_path)
        with open(cache_file, "r", encoding="utf-8") as f:
            datasets = json.loads(f.read())
        knowledge_set = {}
        knowledge_set['train'] = {}
        knowledge_set['valid'] = {}
        knowledge_set['test'] = {}
        if knowledge_type == 'full_knowledge':
            knowledge_set_text = generate_plain_knowledge_set(args.max_length_of_text_knowledge,
                                                                 tokenizer, datasets, logger)
            knowledge_set['train']['text'] = knowledge_set_text['train'] if 'train' in knowledge_set_text else []
            knowledge_set['valid']['text'] = knowledge_set_text['valid'] if 'valid' in knowledge_set_text else []
            knowledge_set['test']['text'] = knowledge_set_text['test'] if 'test' in knowledge_set_text else []
            knowledge_set_infobox = generate_infobox_knowledge_set(args.max_length_of_infobox_knowledge,
                                                                      tokenizer, datasets, logger)
            knowledge_set['train']['infobox'] = knowledge_set_infobox['train'] if 'train' in knowledge_set_infobox else []
            knowledge_set['valid']['infobox'] = knowledge_set_infobox['valid'] if 'valid' in knowledge_set_infobox else []
            knowledge_set['test']['infobox'] = knowledge_set_infobox['test'] if 'test' in knowledge_set_infobox else []

            knowledge_set_graph = generate_graph_knowledge_set(args.max_length_of_graph_knowledge,
                                                                  tokenizer, datasets, logger)
            knowledge_set['train']['graph'] = knowledge_set_graph['train'] if 'train' in knowledge_set_graph else []
            knowledge_set['valid']['graph'] = knowledge_set_graph['valid'] if 'valid' in knowledge_set_graph else []
            knowledge_set['test']['graph'] = knowledge_set_graph['test'] if 'test' in knowledge_set_graph else []
        elif knowledge_type == 'text_knowledge':
            knowledge_set_text = generate_plain_knowledge_set(args.max_length_of_text_knowledge,
                                                                 tokenizer, datasets, logger)
            knowledge_set['train']['text'] = knowledge_set_text['train'] if 'train' in knowledge_set_text else []
            knowledge_set['valid']['text'] = knowledge_set_text['valid'] if 'valid' in knowledge_set_text else []
            knowledge_set['test']['text'] = knowledge_set_text['test'] if 'test' in knowledge_set_text else []
        elif knowledge_type == 'infobox_knowledge':
            knowledge_set_infobox = generate_infobox_knowledge_set(args.max_length_of_infobox_knowledge,
                                                                      tokenizer, datasets, logger)
            knowledge_set['train']['infobox'] = knowledge_set_infobox['train'] if 'train' in knowledge_set_infobox else []
            knowledge_set['valid']['infobox'] = knowledge_set_infobox['valid'] if 'valid' in knowledge_set_infobox else []
            knowledge_set['test']['infobox'] = knowledge_set_infobox['test'] if 'test' in knowledge_set_infobox else []
        elif knowledge_type == 'graph_knowledge':
            knowledge_set_graph = generate_graph_knowledge_set(args.max_length_of_graph_knowledge,
                                                                  tokenizer, datasets, logger)
            knowledge_set['train']['graph'] = knowledge_set_graph['train'] if 'train' in knowledge_set_graph else []
            knowledge_set['valid']['graph'] = knowledge_set_graph['valid'] if 'valid' in knowledge_set_graph else []
            knowledge_set['test']['graph'] = knowledge_set_graph['test'] if 'test' in knowledge_set_graph else []

        torch.save(knowledge_set, knowledge_cache)

    return dataset_src, dataset_tgt, knowledge_set

def tokenize(obj, tokenizer):
    if isinstance(obj, str):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
    if isinstance(obj, dict):
        return dict((n, tokenize(o, tokenizer)) for n, o in obj.items())
    return list(tokenize(o, tokenizer) for o in obj)

def generate_context_dataset(tokenizer, datasets, dataset_cache, logger, src_or_tgt):
    saved_file = {}
    for data_type, dataset in datasets.items():
        if src_or_tgt == 'src':
            sequence = dataset[4]
        else:
            sequence = dataset[5]
        result = []
        for sentence in sequence:
            sentence = sentence.split()
            words = []
            for word in sentence:
                for chars in word:
                    words.append(chars)
            result.append(' '.join(words))

        data = result

        logger.info("Tokenize and encode the dataset")
        logger.info('context %d' % len(data))
        data = tokenize(data, tokenizer)
        saved_file[data_type] = data
    torch.save(saved_file, dataset_cache + '.' + src_or_tgt)

    return saved_file

def generate_plain_knowledge_set(max_length, tokenizer, datasets, logger):
    saved_file = {}
    for data_type, dataset in datasets.items():
        sequence = dataset[1]
        result = []
        for case in sequence:
            if case == '':
                result.append([])
                continue
            sentences = []
            for i in case.split('\t'):
                for sen in i.split('。'):
                    sentences.append(sen)
            one_knowledge = []
            for sentence in sentences:
                if sentence == '':
                    continue
                words = []
                for word in sentence:
                    for chars in word:
                        if word == ' ':
                            continue
                        words.append(chars)
                one_knowledge.append(' '.join(words))
            result.append(one_knowledge[:max_length])
        logger.info("Tokenize and encode the dataset")
        logger.info('text %d' % len(result))
        knowledge_set = tokenize(result, tokenizer)
        saved_file[data_type] = knowledge_set

    return saved_file

def generate_infobox_knowledge_set(max_length, tokenizer, datasets, logger):
    saved_file = {}
    trans = {'category': '类别', 'caption': '说明', 'general': '概括'}
    for data_type, dataset in datasets.items():
        sequence = dataset[2]
        result = []
        for case in sequence:
            case = case.split()
            if len(case) <= 1:
                result.append([])
                continue
            info_knowledge = []
            for kv in case[1:]:
                key = kv.split('_')[0]
                if key == '':
                    continue
                if key[0] == '<':
                    key = key[1:-1]
                if key in trans:
                    key = trans[key]
                value = kv.split(':')[-1]
                key_chars = []
                for word in key:
                    for chars in word:
                        if word == ' ':
                            continue
                        key_chars.append(chars)
                key = ' '.join(key_chars)
                value_chars = []
                for word in value:
                    for chars in word:
                        if word == ' ':
                            continue
                        value_chars.append(chars)
                value = ' '.join(value_chars)
                info_knowledge.append([key, value])
            result.append(info_knowledge[:max_length])
        logger.info("Tokenize and encode the dataset")
        logger.info('infobox %d' % len(result))
        knowledge_set = tokenize(result, tokenizer)

        saved_file[data_type] = knowledge_set

    return saved_file

def generate_graph_knowledge_set(max_length, tokenizer, datasets, logger):
    trans = {'RelatedTo': '相关', 'Synonym': '同义词', 'PartOf': '部分', 'Causes': '引起',
             'MotivatedByGoal': '由目标激发', 'CausesDesire': '引起欲望', 'CapableOf': '能够', 'Desires': '渴望',
             'HasSubevent': '有子事件', 'HasA': '有一个', 'AtLocation': '位于', 'NotDesires': '不渴望',
             'HasContext': '有上下文', 'Antonym': '反义词', 'HasFirstSubevent': '有第一个子事件', 'IsA': '是一个',
             'SymbolOf': '象征', 'UsedFor': '用来', 'MadeOf': '制成', 'HasProperty': '有属性',
             'DerivedFrom': '源自', 'FormOf': '形式', 'SimilarTo':'类似', 'DistinctFrom':'区别于',
             'EtymologicallyDerivedFrom': '词源来自', 'EtymologicallyRelatedTo': '词源相关'}
    saved_file = {}
    for data_type, dataset in datasets.items():
        sequence = dataset[3]
        graph_vocab = dataset[0]
        result = []
        for case in sequence:
            case = case.split()
            if len(case) <= 1:
                result.append([])
                continue
            graph_knowledge = []
            tmp_len = 0
            for triple in case:
                if triple == '0':
                    continue
                tri = []
                h_r_t = graph_vocab[int(triple)].split()
                head, rel, tail = h_r_t[-3], h_r_t[-2], h_r_t[-1]
                if rel in trans:
                    rel = trans[rel]
                else:
                    print('WARN: UNUSED RELATION: ' + rel)
                words = []
                for word in head:
                    for chars in word:
                        if word == ' ':
                            continue
                        words.append(chars)
                tri.append(' '.join(words))
                words = []
                for word in rel:
                    for chars in word:
                        if word == ' ':
                            continue
                        words.append(chars)
                tri.append(' '.join(words))
                words = []
                for word in tail:
                    for chars in word:
                        if word == ' ':
                            continue
                        words.append(chars)
                tri.append(' '.join(words))
                graph_knowledge.append(tri)
                tmp_len += len(tri)
                if tmp_len > 400:
                    break
            result.append(graph_knowledge[:max_length])
        logger.info("Tokenize and encode the dataset")
        logger.info('graph %d' % len(result))
        knowledge_set = tokenize(result, tokenizer)
        saved_file[data_type] = knowledge_set

    return saved_file

def build_dataloaders(args, tokenizer, logger):
    logger.info("Build train and validation dataloaders")

    # test_dataset用于测试
    datasets_src, datasets_tgt, knowledge_sets = get_data(args, tokenizer, args.data_path, args.dataset_cache,
                                                          args.knowledge_cache, logger, args.external_knowledge_type)
    # 用于测试
    test_dataset = WBDataset(args, datasets_src['test'], datasets_tgt['test'], knowledge_sets['test'], tokenizer)
    train_dataset = WBDataset(args, datasets_src['train'], datasets_tgt['train'], knowledge_sets['train'], tokenizer)
    valid_dataset = WBDataset(args, datasets_src['valid'], datasets_tgt['valid'], knowledge_sets['valid'], tokenizer)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None
    train_loader = DataLoader(train_dataset,
                              sampler=train_sampler,
                              collate_fn=train_dataset.collate,
                              num_workers=args.num_workers,
                              batch_size=args.train_batch_size,
                              shuffle=(not args.distributed))
    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler,
                              collate_fn=valid_dataset.collate,
                              num_workers=args.num_workers,
                              batch_size=args.valid_batch_size,
                              shuffle=False)

    return train_loader, valid_loader, train_sampler, valid_sampler


def build_dist_loaders(args, tokenizer, logger):
    logger.info("Build train and validation dataloaders")

    train_dataset = WBdistDataset(tokenizer, data_path=args.train_path)
    valid_dataset = WBdistDataset(tokenizer, data_path=args.valid_path)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None
    train_loader = DataLoader(train_dataset,
                              collate_fn=train_dataset.collate,
                              pin_memory=(args.device == "cuda"),
                              num_workers=args.num_workers,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              shuffle=(not args.distributed))
    valid_loader = DataLoader(valid_dataset,
                              collate_fn=valid_dataset.collate,
                              pin_memory=(args.device == "cuda"),
                              num_workers=args.num_workers,
                              sampler=valid_sampler,
                              batch_size=args.valid_batch_size,
                              shuffle=False)
    return train_loader, valid_loader, train_sampler, valid_sampler
