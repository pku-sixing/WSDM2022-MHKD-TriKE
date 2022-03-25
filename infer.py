# # Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import math
import json
import logging
import random
from argparse import ArgumentParser
from pprint import pformat
from itertools import chain
from tqdm import tqdm

import torch
import torch.nn.functional as F

from transformers import OpenAIGPTLMHeadModel, GPT2LMHeadModel, BertTokenizer

from od.inputters.dataset_wb import WBDataset
from od.inputters.inputter import get_data


SPECIAL_TOKENS = ["[CLS]", "[SEP]", "[PAD]", "[speaker1]", "[speaker2]", '[text]', '[/text]',
                  '[box]', '[/box]', '[graph]', '[/graph]',
                  '[TEXT_SEP]', '[KV_SEP]', '[HR_SEP]', '[TR_SEP]',
                  '[field]', '[/field]', '[triple]', '[/triple]', '[none_knowledge]',
                  '[text_knowledge_mark]', '[infobox_knowledge_mark]', '[graph_knowledge_mark]']
MODEL_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]


def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def build_input_from_segments(history, reply, tokenizer, with_eos=True):
    """ Build a sequence of input from 3 segments: persona, history and last reply """
    bos, eos, pad, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    sequence = [[bos]] + history + [reply + ([eos] if with_eos else [])]
    sequence = [sequence[0]] + [[speaker2 if i % 2 else speaker1] + s for i, s in enumerate(sequence[1:])]
    instance = {}
    instance["input_ids"] = list(chain(*sequence))
    instance["token_type_ids"] = [bos] + [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence[1:])
                                          for _ in s]
    return instance, sequence


def test_data(args):
    with open(args.datapath, "r", encoding="utf-8") as f:
        dataset = json.loads(f.read())
    if isinstance(dataset, dict):
        dataset = dataset["test"]
    return dataset


def sample_sequence(instance, tokenizer, model, args, current_output=None):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    assert SPECIAL_TOKENS[4] == '[speaker2]'
    if current_output is None:
        current_output = []

    for i in range(args.max_length):
        index = len(instance["token_type_ids"]) - 1
        while index >= 0:
            if instance["lm_labels"][index] == -1:
                break
            index -= 1
        input_ids = torch.tensor(instance["input_ids"][:index + 1] + current_output, dtype=torch.long, device=args.device).unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"][:index + 1] + [special_tokens_ids[4]] * len(current_output) , dtype=torch.long, device=args.device).unsqueeze(0)

        logits, *_ = model(input_ids, token_type_ids=token_type_ids)
        logits = logits[0, -1, :] / args.temperature
        logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        probs = F.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
        if i < args.min_length and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())

    return current_output


def main():
    parser = ArgumentParser()
    parser.add_argument('--gpt2', action='store_true', help="use gpt2")
    parser.add_argument("--data_path", type=str, default="data/hke_decode_demo.json", help="Path of the dataset.")
    parser.add_argument("--out_path", type=str, default="", help="Path of response generated.")
    parser.add_argument("--model_checkpoint", type=str, default="models/CDial-GPT2_LCCC-base", help="Path, url or short name of the model")
    parser.add_argument("--dataset_cache", type=str, default="dataset_cache_mske",
                        help="Path or url of the dataset cache")
    parser.add_argument("--knowledge_cache", type=str, default="full_knowledge",
                        help="Path or url of the knowledge cache")
    parser.add_argument("--knowledge_mask", type=str, default="",
                        help="knowledge_mask")
    parser.add_argument('--external_knowledge_type', type=str, default='full_knowledge',
                        choices=['full_knowledge', 'text_knowledge', 'infobox_knowledge', 'graph_knowledge', 'None_knowledge'],
                        help="Select which type of knowledge to use: Text/Infobox/Graph/None")
    parser.add_argument("--device", type=str, default= "cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument('--max_length_of_text_knowledge', type=int, default=512, help="max_length_of_text_knowledge")
    parser.add_argument('--max_length_of_infobox_knowledge', type=int, default=512, help="max_length_of_infobox_knowledge")
    parser.add_argument('--max_length_of_graph_knowledge', type=int, default=512, help="max_length_of_graph_knowledge")
    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--max_length", type=int, default=30, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--temperature", type=float, default=1, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.0,
                        help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.info(pformat(args))

    if args.model_checkpoint == "":
        logging.error("Checkpoint needed!")
        return

    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    logger.info("Get pretrained model and tokenizer")
    tokenizer_class = BertTokenizer
    model_class = OpenAIGPTLMHeadModel if not args.gpt2 else GPT2LMHeadModel
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint, do_lower_case=True)
    model = model_class.from_pretrained(args.model_checkpoint)

    model.to(args.device)
    model.eval()

    def tokenize(obj):
        if isinstance(obj, str):
            return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
        if isinstance(obj, dict):
            return dict((n, tokenize(o)) for n, o in obj.items())
        return list(tokenize(o) for o in obj)

    # dataset = test_data(args)
    datasets_src, datasets_tgt, knowledge_sets = get_data(args, tokenizer, args.data_path, args.dataset_cache, args.knowledge_cache, logger,
                                                          args.external_knowledge_type)
    dataset = WBDataset(args, datasets_src['test'], datasets_tgt['test'], knowledge_sets['test'], tokenizer)

    predictions = []
    for instance in tqdm(dataset, mininterval=1):
        with torch.no_grad():
            out_ids = sample_sequence(instance, tokenizer, model, args)
        out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
        predictions.append(out_text)

    with open(args.out_path, 'w+', encoding="UTF-8") as f:
        f.write("\n".join(predictions))


if __name__ == "__main__":
    main()
