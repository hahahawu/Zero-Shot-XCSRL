
def load_vocab(path, symbol_idx=None):
    if symbol_idx is None:
        symbol_idx = {}
    with open(path, 'r', encoding="utf-8") as fr:
        for symbol in fr:
            symbol = symbol.strip()
            if symbol not in symbol_idx:
                symbol_idx[symbol] = len(symbol_idx)
    return symbol_idx


def gen_labeled_data(in_path, out_file, explicit_arg=False):
    with open(in_path, "r", encoding="utf-8") as fr, open(out_file, "w", encoding="utf-8") as fw:
        for line in fr:
            words, labels = line.strip().split("|||")
            words = words.strip().split(" ")
            pred, words = int(words[0]), words[1:]
            labels = labels.strip().split(" ")
            assert len(words) == len(labels)
            assert labels[pred] == 'B-V'
            new_label = []
            for la in labels:
                if la[2:] in allow_arguments:
                    if explicit_arg:
                        if la not in target_labels:
                            la = la.replace("ARGM-", "").replace("C-", "")
                            if la[2:5] == "ARG":
                                la = la[:2] + "ARG"
                        assert la in target_labels
                        new_label.append(la)
                    else:
                        if la[2:] == 'V':
                            new_label.append(la)
                        else:
                            new_label.append(la[:2] + "ARG")
                else:
                    new_label.append("O")
            fw.write("|||".join([str(pred), " ".join(words), " ".join(new_label)]) + "\n")


if __name__ == '__main__':
    allow_arguments = ["V", "ARG0", "ARG1", "ARGM-LOC", "ARGM-TMP", "ARGM-PRP"]

    target_labels = ["B-V", "I-V", "B-ARG", "I-ARG", "B-LOC", "I-LOC", "B-TMP", "I-TMP", "B-PRP", "I-PRP"]
    for lang in ["zh", "en"]:
        for mode in ["dev", "train"]:
            gen_labeled_data("{}/conll2012.{}.txt".format(lang, mode), "{}.{}.srl".format(lang, mode),
                             explicit_arg=True)
