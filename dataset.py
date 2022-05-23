from torch.utils.data import Dataset
import json
import os
import pickle
import random
import copy


def load_CSRL_data(path):
    instances = []
    with open(path, "r", encoding="utf-8") as fr:
        for line in fr:
            line = line.strip().split("|||")
            sent_id, pred_idx, sent, turn_ids, label = line
            pred_idx = [int(p) for p in pred_idx.split()]
            words = sent.split()
            turn_ids = [int(_) for _ in turn_ids.split()]
            label = label.split()
            instances.append({"sent_id": sent_id, "pred_idx": pred_idx, "words": words, "turn_ids": turn_ids,
                              "label": label})
    return instances


def load_Dialogue_data(path):
    cache_ckpt_path = path + ".ckpt"
    if os.path.exists(cache_ckpt_path):
        return pickle.load(open(cache_ckpt_path, "rb"))
    else:
        instances = []
        with open(path, "r", encoding="utf-8") as fr:
            for line in fr:
                jo = json.loads(line.strip())
                conversations = jo["conversation"]
                if len([_ for _ in conversations if len(_.strip()) > 0]) <= 2:
                    continue
                total_len = 0
                new_conv = []
                for i in range(len(conversations)):
                    if total_len + len(conversations[i]) <= 400:
                        new_conv.append(conversations[i])
                        total_len += len(conversations[i])
                    else:
                        break
                instances.append(new_conv)
            print("There are totally {} dialogues.".format(len(instances)))
            pickle.dump(instances, open(cache_ckpt_path, "wb"))
    return instances


class CSRLData(Dataset):
    def __init__(self, path):
        super(CSRLData, self).__init__()
        self.instances = load_CSRL_data(path)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        return self.instances[idx]


class DialogSet(Dataset):
    def __init__(self, paths):
        super(DialogSet, self).__init__()
        if isinstance(paths, str):
            paths = [paths]
        self.instances = []
        for path in paths:
            self.instances.extend(load_Dialogue_data(path))

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        return self.instances[index]


def load_unpaired_data(path):
    cache_ckpt_path = path + ".ckpt"
    if os.path.exists(cache_ckpt_path):
        return pickle.load(open(cache_ckpt_path, "rb"))
    else:
        instances = []
        with open(path, "r", encoding="utf-8") as fr:
            for line in fr:
                line = line.rstrip("\n")
                instances.append([line])
            print("There are totally {} cross-lingual sentences.".format(len(instances)))
            pickle.dump(instances, open(cache_ckpt_path, "wb"))
        return instances


def load_paired_data(path):
    cache_ckpt_path = path + ".ckpt"
    if os.path.exists(cache_ckpt_path):
        cache = pickle.load(open(cache_ckpt_path, "rb"))
        return cache["sentences"], cache["grams"]
    else:
        sentences = []
        one_gram_hpsi, two_gram_hpsi, three_gram_hpsi, four_gram_hpsi = [], [], [], []
        with open(path, "r", encoding="utf-8") as fr:
            for line in fr:
                jo = json.loads(line.strip())
                en_sent, zh_sent = jo["pair"]
                sentences.append([en_sent.split(" "), zh_sent.split()])
                one_gram_hpsi.append(jo["1_gram"]['en'] + jo["1_gram"]['zh'])
                two_gram_hpsi.append(jo["2_gram"]['en'] + jo["1_gram"]['zh'])
                three_gram_hpsi.append(jo["3_gram"]['en'] + jo["1_gram"]['zh'])
                four_gram_hpsi.append(jo["4_gram"]['en'] + jo["1_gram"]['zh'])
            print("There are totally {} cross-lingual pairs.".format(len(sentences)))
            pickle.dump({"sentences": sentences, "grams": [
                one_gram_hpsi, two_gram_hpsi, three_gram_hpsi, four_gram_hpsi]}, open(cache_ckpt_path, "wb"))
            return sentences, [one_gram_hpsi, two_gram_hpsi, three_gram_hpsi, four_gram_hpsi]


class CrossLingualSet(Dataset):
    def __init__(self, path, use_aligned_data=False):
        super(CrossLingualSet, self).__init__()
        self.instances = []
        if type(path) == str:
            path = [path]
        for p in path:
            if use_aligned_data:
                self.instances, self.n_gram_hpsi = load_paired_data(p)
            else:
                self.instances.extend(load_unpaired_data(p))
        self.alignment = use_aligned_data

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, x):
        if self.alignment:
            pos_pair = self.instances[x]
            # we select negative pairs from 1-4 gram similar pairs and perturbed sequence
            # 0-3 -> 1-4 gram similarity; 4 -> reorder; 5 -> delete words
            op_flag = False
            rd = random.randint(0, 5)
            if rd == 4 and len(pos_pair[0]) >= 8 and len(pos_pair[1]) >= 8:
                negative_pair = copy.deepcopy(pos_pair)
                for sent in negative_pair:
                    rd_s = random.randint(0, len(sent) - 2)
                    rd_e = random.randint(rd_s + 1, len(sent) - 1)
                    _tmp = sent[rd_s:min(len(sent), rd_e + 1)]
                    random.shuffle(_tmp)
                    sent[rd_s:min(len(sent), rd_e + 1)] = _tmp
            elif rd == 5 and len(pos_pair[0]) >= 8 and len(pos_pair[1]) >= 8:
                negative_pair = copy.deepcopy(pos_pair)
                for x, words in enumerate(negative_pair):
                    del_indices = random.choices(range(0, len(words)), k=int(0.15 * len(words)))
                    new_words = []
                    for i in range(len(words)):
                        if i in del_indices:
                            continue
                        else:
                            new_words.append(words[i])
                    negative_pair[x] = new_words
            else:
                op_flag = True
                if rd >= 4:
                    rd = random.randint(0, 3)
                try:
                    gram_hpsi = self.n_gram_hpsi[rd][x]
                    negative_pair = self.instances[int(random.choice(gram_hpsi))]
                except IndexError:
                    print("Index error occurred!")
                    negative_pair = pos_pair
            return pos_pair, negative_pair, op_flag
        else:
            return self.instances[x]


def load_srl_data(path):
    cpkt_cache_file = path + ".ckpt"
    if os.path.exists(cpkt_cache_file):
        return pickle.load(open(cpkt_cache_file, "rb"))
    else:
        instances = []
        with open(path, "r", encoding="utf-8") as fr:
            for line in fr:
                line = line.strip().split("|||")
                pred, words, labels = int(line[0]), line[1].split(" "), line[2].split(" ")
                instances.append((pred, words, labels))
            print("There are totally {} standard SRL instances in file {}.".format(len(instances), path))
            pickle.dump(instances, open(path + ".ckpt", "wb"))
        return instances


class SRLSet(Dataset):
    def __init__(self, file_paths, lang):
        super(SRLSet, self).__init__()
        self.instances = []
        if lang != 'zh+en':
            file_paths = [_ for _ in file_paths if lang in _]
        for file_path in file_paths:
            self.instances.extend(load_srl_data(file_path))

    def __getitem__(self, x):
        return self.instances[x]

    def __len__(self):
        return len(self.instances)
