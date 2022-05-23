import torch
import random
from transformers import AutoTokenizer
from transformers.models.xlm_roberta.tokenization_xlm_roberta import SPIECE_UNDERLINE
import copy
import re

from torch.nn.utils.rnn import pad_sequence


def load_vocab(path, symbol_idx=None):
    if symbol_idx is None:
        symbol_idx = {}
    with open(path, 'r', encoding="utf-8") as fr:
        for symbol in fr:
            symbol = symbol.strip()
            if symbol not in symbol_idx:
                symbol_idx[symbol] = len(symbol_idx)
    return symbol_idx


class Processor(object):
    def __init__(self, task, pretrain_model_name, label_path=None, max_seq_len=512, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrain_model_name, use_fast=True)
        self.tokenizer.add_tokens(["[human]", "[agent]"])
        # self.tokenizer.add_special_tokens({'bos_token': "[BOS]"})
        self.max_sequence_len = min(self.tokenizer.model_max_length, max_seq_len)
        if task == 'csrl':
            label_vocab = {"O": 0}
            self.label_vocab = load_vocab(label_path, label_vocab)
        elif task == 'srl':
            self.label_vocab = {"O": 0}
            for la in ["B-V", "I-V", "B-ARG", "I-ARG", "B-LOC", "I-LOC", "B-TMP", "I-TMP", "B-PRP", "I-PRP"]:
                self.label_vocab[la] = len(self.label_vocab)
        elif task == 'dialogue':
            self.is_reorder = kwargs["is_reorder"]
            self.is_role_detection = kwargs["is_role_detection"]
        elif task == 'language_model':
            self.word_vocab = self.tokenizer.get_vocab()
            self.alignment = kwargs["alignment"]
        else:
            raise ValueError("unexpected task {}".format(task))

        self.pad_token_id = self.tokenizer.pad_token_id

    def encode_csrl_case(self, case):
        words, pred_idx, turn_ids, label = case["words"], case["pred_idx"], case["turn_ids"], case["label"]

        subword_indices = []
        speaker_role_ids = []
        tokens = []
        word_idx = 2
        speaker_role = "[agent]"
        speaker_role_id_mapping = {"[agent]": 1, "[human]": 2}
        for idx, word in enumerate(words):
            tokenized_words = self.tokenizer.tokenize(word)
            tokenized_words = [t for t in tokenized_words if t != SPIECE_UNDERLINE]
            if len(tokenized_words) + len(tokens) >= self.max_sequence_len - 3:
                words = words[:idx]
                break
            tokens.extend(tokenized_words)
            subword_indices.extend([word_idx] * len(tokenized_words))
            word_idx += 1

            if word in ["[agent]", "[human]"]:
                speaker_role = word
            speaker_role_ids.append(speaker_role_id_mapping[speaker_role])

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        sequence_length = len(words) + 2

        # add [CLS] and [SEP]
        dialogue_indices, utterance_num = self.__get_dialogue_indices(words)
        subword_indices.insert(0, 1)  # [CLS]
        speaker_role_ids.append(0)
        subword_indices.append(subword_indices[-1] + 1)  # [SEP]
        speaker_role_ids.append(0)
        subword_indices.append(0)  # [PAD]

        input_ids.insert(0, self.tokenizer.cls_token_id)  # [CLS]
        input_ids.append(self.tokenizer.sep_token_id)  # [SEP]
        input_ids.append(self.tokenizer.pad_token_id)  # [PAD]

        pred_idx = [p + 1 for p in pred_idx]
        # pred_idx += (self.max_predicate_length - len(pred_idx)) * [0]
        predicate_ids = [2 if _ in pred_idx else 1 for _ in range(sequence_length)]

        turn_ids = turn_ids[:sequence_length - 2]
        turn_ids = [0] + [min(_ + 1, 11) for _ in turn_ids] + [0]
        label = label[:sequence_length - 2]
        label = ["O"] + label + ["O"]
        label = [self.label_vocab[la] for la in label]

        tokenized_sequence_length = len(input_ids) - 1

        assert len(input_ids) == len(subword_indices) == tokenized_sequence_length + 1
        assert len(label) == sequence_length == len(turn_ids) == len(dialogue_indices) == len(
            speaker_role_ids) == len(predicate_ids)

        return {
            "word_ids": torch.as_tensor(input_ids),
            "predicate_ids": torch.as_tensor(predicate_ids),
            "turn_ids": torch.as_tensor(turn_ids),
            "speaker_role_ids": torch.as_tensor(speaker_role_ids),
            "subword_indices": torch.as_tensor(subword_indices),
            "sequence_length": sequence_length,
            "tokenized_sequence_length": tokenized_sequence_length,
            "dialogue_indices": torch.as_tensor(dialogue_indices),
            "utterance_num": utterance_num,
            "label": torch.as_tensor(label)
        }

    def __dialogue_processing(self, conversations):
        words = []
        for conv in conversations:
            words.append(self.tokenizer.bos_token)
            words.extend(conv.split(" "))
        return words

    def __dialogue_tokenization(self, conversations, labels):
        words = self.__dialogue_processing(conversations)
        cur_turn = 0
        subword_indices = []
        tokens = []
        position_ids = []
        cur_pos = 0
        word_idx = 2
        for idx, word in enumerate(words):
            if word == self.tokenizer.eos_token:
                cur_turn += 1
                cur_pos = 0
            tokenized_words = self.tokenizer.tokenize(word)
            tokenized_words = [t for t in tokenized_words if t != SPIECE_UNDERLINE]
            if len(tokenized_words) + len(tokens) >= self.max_sequence_len - 3:
                words = words[:idx]
                labels = labels[:cur_turn]
                break
            tokens.extend(tokenized_words)
            position_ids.extend(list(range(cur_pos, cur_pos+len(tokenized_words))))
            subword_indices.extend([word_idx] * len(tokenized_words))
            word_idx += 1
            cur_pos += len(tokenized_words)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        dialogue_indices, utterance_num = self.__get_dialogue_indices(words)
        labels.insert(0, -100)
        labels.append(-100)

        subword_indices.insert(0, 1)  # [CLS]
        subword_indices.append(subword_indices[-1] + 1)  # [SEP]
        subword_indices.append(0)  # add [PAD]

        input_ids.insert(0, self.tokenizer.cls_token_id)  # [CLS]
        input_ids.append(self.tokenizer.sep_token_id)  # [SEP]
        input_ids.append(self.pad_token_id)  # [PAD]

        position_ids.insert(0, 0)
        position_ids.append(0)
        position_ids.append(0)

        sequence_length = len(words) + 2
        tokenized_sequence_length = len(input_ids) - 1

        assert len(input_ids) == len(subword_indices) == len(position_ids)
        assert len(input_ids) <= self.tokenizer.model_max_length
        assert len(dialogue_indices) == len(words) + 2
        assert sequence_length == max(subword_indices)

        return input_ids, subword_indices, dialogue_indices, utterance_num, labels, \
               sequence_length, tokenized_sequence_length, position_ids

    def encode_dialog_case(self, case):
        conversations = case
        conversations = [_ for _ in conversations if len(_.strip()) > 0]

        reorder_conv = copy.deepcopy(conversations)
        speaker_conv = copy.deepcopy(conversations)

        return_dict = {}

        # utterance reorder
        if self.is_reorder:
            k = 0.4  # To reduce the search space, we only reorder k*U utterances
            num_reorder_utterance = max(min(int(k * len(reorder_conv)), 10), 2)
            # rand_s = random.randint(0, max(len(reorder_conv) - 1 - num_reorder_utterance, 0))
            _tmp = conversations[-num_reorder_utterance:]
            _tmp_seq = list(range(0, len(_tmp)))
            random.shuffle(_tmp_seq)
            _new_tmp = []
            for _t in _tmp_seq:
                _new_tmp.append(_tmp[_t])
            reorder_conv[-num_reorder_utterance:] = _new_tmp
            reorder_label = [-100] * (len(reorder_conv) - num_reorder_utterance) + _tmp_seq
            assert len(reorder_label) == len(reorder_conv)

            reorder_input_ids, reorder_subword_indices, reorder_dialogue_indices, reorder_utterance_num, reorder_label, \
            reorder_sequence_length, reorder_tokenized_sequence_length, reorder_position_ids = \
                self.__dialogue_tokenization(reorder_conv, reorder_label)
            return_dict["reorder_input_ids"] = torch.as_tensor(reorder_input_ids)
            return_dict["reorder_subword_indices"] = torch.as_tensor(reorder_subword_indices)
            return_dict["reorder_dialogue_indices"] = torch.as_tensor(reorder_dialogue_indices)
            return_dict["reorder_utterance_num"] = reorder_utterance_num
            return_dict["reorder_label"] = torch.as_tensor(reorder_label)
            return_dict["reorder_sequence_length"] = reorder_sequence_length
            return_dict["reorder_tokenized_sequence_length"] = reorder_tokenized_sequence_length
            return_dict["reorder_position_ids"] = torch.as_tensor(reorder_position_ids)

        # speaker role identification; split the utterance by punctuations
        if self.is_role_detection:
            # we cut an utterance into 5 sub-sentences at maximum.
            max_sub_sentences = 5
            new_role_ids, new_speaker_conv = [], []
            for idx, conv in enumerate(speaker_conv):
                # 50% utterance unchanged and 50% utterances are cut
                if random.random() < 0.5:
                    split_sub_sentences = re.split('[.,?!。，？！]', conv)
                    split_sub_sentences = [_.strip() for _ in split_sub_sentences if len(_.strip()) > 0]
                    if len(split_sub_sentences) > max_sub_sentences:
                        split_sub_sentences[max_sub_sentences - 1] = " ".join(split_sub_sentences[max_sub_sentences:])
                    for split_sent in split_sub_sentences:
                        new_role_ids.append(idx % 2)
                        new_speaker_conv.append(split_sent)
                else:
                    new_role_ids.append(idx % 2)
                    new_speaker_conv.append(conv)
            assert len(new_role_ids) == len(new_speaker_conv)
            role_input_ids, role_subword_indices, role_dialogue_indices, role_utterance_num, new_role_ids, \
            role_sequence_length, role_tokenized_sequence_length, role_position_ids = self.__dialogue_tokenization(
                new_speaker_conv, new_role_ids)
            return_dict["role_input_ids"] = torch.as_tensor(role_input_ids)
            return_dict["role_subword_indices"] = torch.as_tensor(role_subword_indices)
            return_dict["role_dialogue_indices"] = torch.as_tensor(role_dialogue_indices)
            return_dict["role_utterance_num"] = role_utterance_num
            return_dict["role_label"] = torch.as_tensor(new_role_ids)
            return_dict["role_sequence_length"] = role_sequence_length
            return_dict["role_tokenized_sequence_length"] = role_tokenized_sequence_length
            return_dict["role_position_ids"] = torch.as_tensor(role_position_ids)

        return return_dict

    def __tokenize_sequence(self, words):
        subword_indices = []
        tokens = []
        word_idx = 2
        for idx, word in enumerate(words):
            tokenized_words = self.tokenizer.tokenize(word)
            tokenized_words = [t for t in tokenized_words if t != SPIECE_UNDERLINE]
            if len(tokenized_words) + len(tokens) >= self.max_sequence_len - 3:
                words = words[:idx]
                break
            tokens.extend(tokenized_words)
            subword_indices.extend([word_idx] * len(tokenized_words))
            word_idx += 1
        return tokens

    def concat_and_convert_tokens_to_input_ids(self, token1, token2=None):
        if token2 is not None:
            return self.tokenizer.convert_tokens_to_ids(
                [self.tokenizer.cls_token] + token1 + [self.tokenizer.sep_token] + token2 + [self.tokenizer.sep_token])
        else:
            return self.tokenizer.convert_tokens_to_ids(
                [self.tokenizer.cls_token] + token1 + [self.tokenizer.sep_token])

    def encoder_lm_case(self, case):
        if len(case) > 1:
            pos_pair, negative_pair, op_flag = case
            en_words, zh_words = pos_pair
            neg_en_words, neg_zh_words = negative_pair
            en_tokens, neg_en_tokens = self.__tokenize_sequence(en_words), self.__tokenize_sequence(neg_en_words)
            zh_tokens, neg_zh_tokens = self.__tokenize_sequence(zh_words), self.__tokenize_sequence(neg_zh_words)
            en_zh = self.concat_and_convert_tokens_to_input_ids(en_tokens, zh_tokens)
            zh_en = self.concat_and_convert_tokens_to_input_ids(zh_tokens, en_tokens)

            # For psi task, construct [neg, neg] pairs and [pos, neg] pairs
            # [neg, neg] pair
            if op_flag:
                if random.random() < 0.5:
                    pos_pair = self.concat_and_convert_tokens_to_input_ids(neg_en_tokens, neg_zh_tokens)
                else:
                    pos_pair = self.concat_and_convert_tokens_to_input_ids(neg_zh_tokens, neg_en_tokens)
            else:
                if random.random() < 0.5:
                    pos_pair = self.concat_and_convert_tokens_to_input_ids(en_tokens, zh_tokens)
                else:
                    pos_pair = self.concat_and_convert_tokens_to_input_ids(zh_tokens, en_tokens)

            # [pos, neg] pairs
            rd = random.random()
            if rd > 0.75:
                neg_pair = self.concat_and_convert_tokens_to_input_ids(en_tokens, neg_zh_tokens)
            elif rd > 0.5:
                neg_pair = self.concat_and_convert_tokens_to_input_ids(zh_tokens, neg_en_tokens)
            elif rd > 0.25:
                neg_pair = self.concat_and_convert_tokens_to_input_ids(neg_en_tokens, zh_tokens)
            else:
                neg_pair = self.concat_and_convert_tokens_to_input_ids(neg_zh_tokens, en_tokens)

            en_zh_mlm_label, en_zh_ids = self.__masking(en_zh)
            zh_en_mlm_label, zh_en_ids = self.__masking(zh_en)

            assert len(en_zh_mlm_label) == len(en_zh_ids) == len(zh_en_mlm_label) == len(zh_en_ids)

            return {
                "en_zh_ids": torch.as_tensor(en_zh_ids),
                "zh_en_ids": torch.as_tensor(zh_en_ids),
                "sequence_length": len(en_zh_mlm_label),
                "en_zh_label": torch.as_tensor(en_zh_mlm_label),
                "zh_en_label": torch.as_tensor(zh_en_mlm_label),
                "pos_pair": torch.as_tensor(pos_pair),
                "neg_pair": torch.as_tensor(neg_pair),
            }

        elif len(case) == 1:
            words = case[0].split()
            tokens = self.__tokenize_sequence(words)
            input_ids = self.concat_and_convert_tokens_to_input_ids(tokens)
            mask_label, input_ids = self.__masking(input_ids)
            assert len(mask_label) == len(input_ids)
            return {
                "input_ids": torch.as_tensor(input_ids),
                "sequence_length": len(input_ids),
                "mlm_label": torch.as_tensor(mask_label)
            }
        else:
            raise ValueError("Unexpected example length %d", len(case))

    def encode_srl_case(self, case):
        pred_idx, words, labels = case

        subword_indices = []
        tokens = []
        word_idx = 2
        for idx, word in enumerate(words):
            tokenized_words = self.tokenizer.tokenize(word)
            tokenized_words = [t for t in tokenized_words if t != SPIECE_UNDERLINE]
            if len(tokenized_words) + len(tokens) >= self.max_sequence_len - 3:
                words = words[:idx]
                break
            tokens.extend(tokenized_words)
            subword_indices.extend([word_idx] * len(tokenized_words))
            word_idx += 1

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        sequence_length = len(words) + 2

        # add [CLS] and [SEP]
        subword_indices.insert(0, 1)  # [CLS]
        subword_indices.append(subword_indices[-1] + 1)  # [SEP]
        subword_indices.append(0)  # [PAD]
        input_ids.insert(0, self.tokenizer.cls_token_id)  # [CLS]
        input_ids.append(self.tokenizer.sep_token_id)  # [SEP]
        input_ids.append(self.tokenizer.pad_token_id)  # [PAD]

        predicate_ids = [2 if _ == pred_idx + 1 else 1 for _ in range(sequence_length)]
        labels = labels[:sequence_length - 2]
        labels = ["O"] + labels + ["O"]
        labels = [self.label_vocab[la] for la in labels]
        tokenized_sequence_length = len(input_ids) - 1

        assert len(input_ids) == len(subword_indices)
        assert len(labels) == sequence_length == len(predicate_ids)

        return {
            "word_ids": torch.as_tensor(input_ids),
            "predicate_ids": torch.as_tensor(predicate_ids),
            "subword_indices": torch.as_tensor(subword_indices),
            "sequence_length": sequence_length,
            "tokenized_sequence_length": tokenized_sequence_length,
            "label": torch.as_tensor(labels)
        }

    def csrl_collate_fn(self, cases):
        batch_sample = {
            "word_ids": [],
            "predicate_ids": [],
            "turn_ids": [],
            "speaker_role_ids": [],
            "subword_indices": [],
            "sequence_length": [],
            "tokenized_sequence_length": [],
            "dialogue_indices": [],
            "utterance_num": [],
            "label": []
        }
        # max_subword_index = 0
        for case in cases:
            encoded_csrl_example = self.encode_csrl_case(case)
            batch_sample["word_ids"].append(encoded_csrl_example["word_ids"])
            batch_sample["predicate_ids"].append(encoded_csrl_example["predicate_ids"])
            batch_sample["turn_ids"].append(encoded_csrl_example["turn_ids"])
            batch_sample["speaker_role_ids"].append(encoded_csrl_example["speaker_role_ids"])
            batch_sample["subword_indices"].append(encoded_csrl_example["subword_indices"])
            # max_subword_index = max_subword_index if max_subword_index >= encoded_csrl_example["subword_indices"][
            #     -1] else encoded_csrl_example["subword_indices"][-1]
            batch_sample["sequence_length"].append(encoded_csrl_example["sequence_length"])
            batch_sample["tokenized_sequence_length"].append(encoded_csrl_example["tokenized_sequence_length"])
            batch_sample["dialogue_indices"].append(encoded_csrl_example["dialogue_indices"])
            batch_sample["utterance_num"].append(encoded_csrl_example["utterance_num"])
            batch_sample["label"].append(encoded_csrl_example["label"])

        # for x in batch_sample["subword_indices"]:
        #     x[-1] = max_subword_index

        batch_sample["word_ids"] = pad_sequence(batch_sample["word_ids"], batch_first=True,
                                                padding_value=self.pad_token_id)
        batch_sample["predicate_ids"] = pad_sequence(batch_sample["predicate_ids"], batch_first=True, padding_value=0)
        batch_sample["turn_ids"] = pad_sequence(batch_sample["turn_ids"], batch_first=True, padding_value=0)
        batch_sample["speaker_role_ids"] = pad_sequence(batch_sample["speaker_role_ids"], batch_first=True,
                                                        padding_value=0)
        batch_sample["subword_indices"] = pad_sequence(batch_sample["subword_indices"], batch_first=True,
                                                       padding_value=0)
        batch_sample["sequence_length"] = torch.as_tensor(batch_sample["sequence_length"])
        batch_sample["tokenized_sequence_length"] = torch.as_tensor(batch_sample["tokenized_sequence_length"])
        batch_sample["dialogue_indices"] = pad_sequence(batch_sample["dialogue_indices"], batch_first=True,
                                                        padding_value=0)
        batch_sample["utterance_num"] = torch.as_tensor(batch_sample["utterance_num"])
        batch_sample["label"] = pad_sequence(batch_sample["label"], batch_first=True, padding_value=-100)

        return batch_sample

    def srl_collate_fn(self, cases):
        batch_sample = {
            "word_ids": [],
            "predicate_ids": [],
            "subword_indices": [],
            "sequence_length": [],
            "tokenized_sequence_length": [],
            "label": []
        }

        for case in cases:
            encoded_srl_example = self.encode_srl_case(case)
            batch_sample["word_ids"].append(encoded_srl_example["word_ids"])
            batch_sample["subword_indices"].append(encoded_srl_example["subword_indices"])
            batch_sample["predicate_ids"].append(encoded_srl_example["predicate_ids"])
            batch_sample["sequence_length"].append(encoded_srl_example["sequence_length"])
            batch_sample["tokenized_sequence_length"].append(encoded_srl_example["tokenized_sequence_length"])
            batch_sample["label"].append(encoded_srl_example["label"])

        batch_sample["word_ids"] = pad_sequence(batch_sample["word_ids"], batch_first=True,
                                                padding_value=self.pad_token_id)
        batch_sample["subword_indices"] = pad_sequence(batch_sample["subword_indices"], batch_first=True,
                                                       padding_value=0)
        batch_sample["predicate_ids"] = pad_sequence(batch_sample["predicate_ids"], batch_first=True, padding_value=0)
        batch_sample["sequence_length"] = torch.as_tensor(batch_sample["sequence_length"])
        batch_sample["tokenized_sequence_length"] = torch.as_tensor(batch_sample["tokenized_sequence_length"])
        batch_sample["label"] = pad_sequence(batch_sample["label"], batch_first=True, padding_value=-100)

        return batch_sample

    def dialog_collate_fn(self, cases):
        batch_sample = {
            "reorder_input_ids": [],
            "reorder_subword_indices": [],
            "reorder_dialogue_indices": [],
            "reorder_utterance_num": [],
            "reorder_label": [],
            "reorder_sequence_length": [],
            "reorder_tokenized_sequence_length": [],
            "reorder_position_ids": [],
            "role_input_ids": [],
            "role_subword_indices": [],
            "role_dialogue_indices": [],
            "role_utterance_num": [],
            "role_label": [],
            "role_sequence_length": [],
            "role_tokenized_sequence_length": [],
            "role_position_ids": []
        }
        for case in cases:
            encoded_dialogue_example = self.encode_dialog_case(case)
            for k, v in encoded_dialogue_example.items():
                batch_sample[k].append(v)

        if self.is_reorder:
            batch_sample["reorder_input_ids"] = pad_sequence(batch_sample["reorder_input_ids"], batch_first=True,
                                                             padding_value=self.pad_token_id)
            batch_sample["reorder_subword_indices"] = pad_sequence(batch_sample["reorder_subword_indices"],
                                                                   batch_first=True, padding_value=0)
            batch_sample["reorder_dialogue_indices"] = pad_sequence(batch_sample["reorder_dialogue_indices"],
                                                                    batch_first=True, padding_value=0)
            batch_sample["reorder_utterance_num"] = torch.as_tensor(batch_sample["reorder_utterance_num"])
            batch_sample["reorder_label"] = pad_sequence(batch_sample["reorder_label"], batch_first=True, padding_value=-100)
            batch_sample["reorder_sequence_length"] = torch.as_tensor(batch_sample["reorder_sequence_length"])
            batch_sample["reorder_tokenized_sequence_length"] = torch.as_tensor(batch_sample["reorder_tokenized_sequence_length"])
            batch_sample["reorder_position_ids"] = pad_sequence(batch_sample["reorder_position_ids"], batch_first=True,
                                                                padding_value=0)
        if self.is_role_detection:
            batch_sample["role_input_ids"] = pad_sequence(batch_sample["role_input_ids"], batch_first=True,
                                                          padding_value=self.tokenizer.pad_token_id)
            batch_sample["role_subword_indices"] = pad_sequence(batch_sample["role_subword_indices"], batch_first=True,
                                                                padding_value=0)
            batch_sample["role_dialogue_indices"] = pad_sequence(batch_sample["role_dialogue_indices"], batch_first=True,
                                                                 padding_value=0)
            batch_sample["role_utterance_num"] = torch.as_tensor(batch_sample["role_utterance_num"])
            batch_sample["role_label"] = pad_sequence(batch_sample["role_label"], batch_first=True, padding_value=-100)
            batch_sample["role_sequence_length"] = torch.as_tensor(batch_sample["role_sequence_length"])
            batch_sample["role_tokenized_sequence_length"] = torch.as_tensor(batch_sample["role_tokenized_sequence_length"])
            batch_sample["role_position_ids"] = pad_sequence(batch_sample["role_position_ids"], batch_first=True,
                                                             padding_value=0)

        return batch_sample

    def lm_collate_fn(self, cases):
        if self.alignment:
            batch_sample = {
                "input_ids": [],
                "sequence_length": [],
                "mlm_label": [],
                "psi_pair": [],
                "psi_sequence_length": [],
                "psi_label": []
            }
            for case in cases:
                encoded_alignment_example = self.encoder_lm_case(case)
                batch_sample["input_ids"].append(encoded_alignment_example["en_zh_ids"])
                batch_sample["input_ids"].append(encoded_alignment_example["zh_en_ids"])
                batch_sample["sequence_length"].append(encoded_alignment_example["sequence_length"])
                batch_sample["sequence_length"].append(encoded_alignment_example["sequence_length"])
                batch_sample["mlm_label"].append(encoded_alignment_example["en_zh_label"])
                batch_sample["mlm_label"].append(encoded_alignment_example["zh_en_label"])
                batch_sample["psi_pair"].append(encoded_alignment_example["pos_pair"])
                batch_sample["psi_pair"].append(encoded_alignment_example["neg_pair"])
                batch_sample["psi_sequence_length"].append(len(encoded_alignment_example["pos_pair"]))
                batch_sample["psi_sequence_length"].append(len(encoded_alignment_example["neg_pair"]))
                batch_sample["psi_label"].append(1)
                batch_sample["psi_label"].append(0)
            batch_sample["input_ids"] = pad_sequence(batch_sample["input_ids"], batch_first=True,
                                                     padding_value=self.tokenizer.pad_token_id)
            batch_sample["sequence_length"] = torch.as_tensor(batch_sample["sequence_length"])
            batch_sample["mlm_label"] = pad_sequence(batch_sample["mlm_label"], batch_first=True, padding_value=-100)
            batch_sample["psi_pair"] = pad_sequence(batch_sample["psi_pair"], batch_first=True,
                                                    padding_value=self.tokenizer.pad_token_id)
            batch_sample["psi_sequence_length"] = torch.as_tensor(batch_sample["psi_sequence_length"])
            batch_sample["psi_label"] = torch.as_tensor(batch_sample["psi_label"])
        else:
            batch_sample = {
                "input_ids": [],
                "sequence_length": [],
                "mlm_label": []
            }

            for case in cases:
                encoded_alignment_example = self.encoder_lm_case(case)
                batch_sample["input_ids"].append(encoded_alignment_example["input_ids"])
                batch_sample["sequence_length"].append(encoded_alignment_example["sequence_length"])
                batch_sample["mlm_label"].append(encoded_alignment_example["mlm_label"])

            batch_sample["input_ids"] = pad_sequence(batch_sample["input_ids"], batch_first=True,
                                                     padding_value=self.pad_token_id)
            batch_sample["sequence_length"] = torch.as_tensor(batch_sample["sequence_length"])
            batch_sample["mlm_label"] = pad_sequence(batch_sample["mlm_label"], batch_first=True, padding_value=-100)

        return batch_sample

    def __get_dialogue_indices(self, words):
        utterance_num = 0
        # Denote [CLS] as the first utterance
        dialogue_indices = [1]
        utterance_num += 1
        dialogue_idx = 1
        for w in words:
            if w in ["[human]", "[agent]", self.tokenizer.bos_token]:
                dialogue_idx += 1
                dialogue_indices.append(dialogue_idx)
                utterance_num += 1
                cur_pos = 0
            else:
                dialogue_indices.append(dialogue_idx)
        # add [SEP] as the last utterance
        dialogue_indices.append(dialogue_idx + 1)
        utterance_num += 1
        return dialogue_indices, utterance_num

    @staticmethod
    def __preprocessing(conversations, is_truncation):
        words = []
        for idx, conv in enumerate(conversations):
            if idx % 2 == 0:
                tmp_words = "[human] " + conv
                if not is_truncation:
                    conversations[idx] = tmp_words
            else:
                tmp_words = "[agent] " + conv
                if not is_truncation:
                    conversations[idx] = tmp_words
            split_word = tmp_words.split(" ")
            if len(words) + len(split_word) > 500:
                conversations = conversations[:idx]
                break
            else:
                words.extend(split_word)
        return conversations, words

    def __masking(self, token_ids):
        targets = [-100 for _ in range(len(token_ids))]

        for i, token_id in enumerate(token_ids):
            if token_id in [self.tokenizer.cls_token_id, self.tokenizer.sep_token_id]:
                continue
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                if prob < 0.8:
                    token_ids[i] = self.tokenizer.mask_token_id
                elif prob < 0.9:
                    token_ids[i] = random.choice(list(self.word_vocab.values()))

                targets[i] = token_id
        return targets, token_ids

# if __name__ == '__main__':
#     from dataset import load_Dialogue_data
#     from tqdm import tqdm
#     processor = Processor(task='dialogue', pretrain_model_name='xlm-roberta-base', label_path="./data/dialog_data")
#     instances = []
#     for path in ["./data/dialog_data/duconv.train.txt"]:
#         instances.extend(load_Dialogue_data(path))
#     print("total case: ", len(instances))
#     for idx, ins in tqdm(enumerate(instances)):
#         processor.encode_dialog_case(ins)
