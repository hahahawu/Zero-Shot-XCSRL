import math
from collections import Counter, OrderedDict
import json
import os

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR


def fine_print_res(words, labels, reverse_mapping=None):
    role_idx = {}
    for _ in range(len(labels)):
        label = labels[_]
        if label[0] == "B" or label[0] == "I" or label[0] == "V":
            if label[0] == "V":
                role = "V"
            else:
                role = label[2:]
            if role in role_idx:
                role_idx[role].append(_)
            else:
                role_idx[role] = [_]

    # make sure that the role idx list is in the order
    for role in role_idx:
        l = role_idx[role]
        if len(l) > 0:
            fi = l[0]
            li = l[-1]
            if li - fi + 1 != len(l):
                # we need to make sure to select the first occurred span
                if role != "V":
                    new_fi, new_li = is_include_multi_mentions(labels, fi, li)
                    fi = new_fi
                    li = new_li

                l = [t for t in range(fi, li + 1)]
                role_idx[role] = l

    if reverse_mapping is not None:
        for role in role_idx:
            input_left = reverse_mapping[role_idx[role][0]]
            input_right = reverse_mapping[role_idx[role][-1]] + 1
            role_idx[role] = [_ for _ in range(input_left, input_right)]

    sep_symbol = " "
    all_role_spans = {}
    inner_role_spans = {}
    inter_role_spans = {}
    for key in role_idx:
        all_role_spans[key] = sep_symbol.join([words[id] for id in role_idx[key]])

    if "V" not in role_idx or len(role_idx["V"]) == 0:
        inner_role_spans = all_role_spans
        inter_role_spans = all_role_spans
    else:
        pred_idx = role_idx["V"][0]
        for key in role_idx:
            val = sep_symbol.join([words[id] for id in role_idx[key]])
            if key == "V":
                inner_role_spans[key] = val
                inter_role_spans[key] = val
                continue
            tmp = role_idx[key]
            left_idx = tmp[0]
            right_idx = tmp[-1]
            if left_idx > pred_idx:
                inner_role_spans[key] = val
            elif right_idx < pred_idx:
                found_turn_break = False
                for _ in range(right_idx + 1, pred_idx):
                    if words[_] == "[human]" or words[_] == "[agent]":
                        found_turn_break = True
                        break
                if found_turn_break:
                    inter_role_spans[key] = val
                else:
                    inner_role_spans[key] = val

    return all_role_spans, inner_role_spans, inter_role_spans, role_idx


def is_include_multi_mentions(labels, from_idx, to_idx):
    is_find_mentions = False
    new_from_idx = -1
    new_to_idx = -1
    for _ in range(from_idx, to_idx + 1):
        if labels[_][0] == "B":
            if not is_find_mentions:
                is_find_mentions = True
                new_from_idx = _
                new_to_idx = _
            else:
                return new_from_idx, new_to_idx
        elif labels[_][0] == "I":
            new_to_idx = _
    return from_idx, to_idx


def update_counts_intersect(v1, v2, is_token_level):
    if v1 == '' or v2 == '':
        return 0
    if is_token_level:
        v1 = Counter(v1.split())
        v2 = Counter(v2.split())
        res = 0
        for k, cnt1 in v1.items():
            if k in v2:
                res += min(cnt1, v2[k])
        return res
    else:
        return v1 == v2


def update_counts_denominator(conv, is_token_level):
    counts = 0
    for pas in conv:
        for k, v in pas.items():
            if k != 'V':  # don't count "pred" for each PA structure
                counts += len(v.split()) if is_token_level else 1
    return counts


# is_sync: whether ref-file and prd-file have the same content. This is always Ture except
# when the prd-file is after rewriting.
def update_counts(ref_conv, prd_conv, counts, is_sync, is_token_level):
    counts[1] += update_counts_denominator(ref_conv, is_token_level)
    counts[2] += update_counts_denominator(prd_conv, is_token_level)
    if is_sync:
        for ref_pas, prd_pas in zip(ref_conv, prd_conv):
            for k, v1 in ref_pas.items():
                if k == 'V':
                    continue
                v2 = prd_pas.get(k, '')
                counts[0] += update_counts_intersect(v1, v2, is_token_level)
    else:
        for ref_pas in ref_conv:
            for prd_pas in prd_conv:
                if prd_pas['V'] == ref_pas['V']:
                    for k, v1 in ref_pas.items():
                        if k == 'V':
                            continue
                        v2 = prd_pas.get(k, '')
                        counts[0] += update_counts_intersect(v1, v2, is_token_level)
                    break


def calc_f1(ref, prd, is_sync=True, is_token_level=False):
    counts = [0, 0, 0]
    update_counts(ref, prd, counts, is_sync, is_token_level)
    p = 0.0 if counts[2] == 0 else counts[0] / counts[2]
    r = 0.0 if counts[1] == 0 else counts[0] / counts[1]
    f = 0.0 if p == 0.0 or r == 0.0 else 2 * p * r / (p + r)
    return {'P': p, 'R': r, 'F': f}


def save_hparams_dict(hparams, saved_path):
    h_params_dict = vars(hparams)
    json.dump(h_params_dict, open(saved_path, "w", encoding="utf-8"), ensure_ascii=False, indent=4)


def str2bool(v):
    if v.lower() in ('y', 'yes', 't', 'true', '1'):
        return True
    elif v.lower() in ('n', 'no', 'f', 'false', '0'):
        return False


def get_inverse_square_root_schedule_with_warmup(
    optimizer, warmup_steps, last_epoch=-1
):
    """
    Create a schedule with linear warmup and then inverse square root decay.
    Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
    Inverse square root decreases learning rate from 1. to 0. over remaining steps.
    """

    def lr_lambda(step):
        decay_factor = warmup_steps ** 0.5
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return decay_factor * step ** -0.5

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_linear_schedule_with_warmup(
    optimizer, warmup_steps, training_steps, last_epoch=-1
):
    """
    Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        ratio = (training_steps - step) / max(1, training_steps - warmup_steps)
        return max(ratio, 0)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def z_norm(inputs, epsilon=1e-9):
    mean = inputs.mean(0, keepdim=True)
    var = inputs.var(0, unbiased=False, keepdim=True)
    return (inputs - mean) / torch.sqrt(var + epsilon)


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def load_from_pretrained_weights(model, pretrained_encoders):
    pretrained_encoders = pretrained_encoders.split(",")
    new_state_dict = OrderedDict()
    for pretrained_encoder in pretrained_encoders:
        ckpt_dir = "checkpoints" + os.path.sep + pretrained_encoder + os.path.sep + "checkpoints" + \
                   os.path.sep + pretrained_encoder
        ckpt_name = os.listdir(ckpt_dir)
        select_name = ckpt_name[0]
        for name in ckpt_name:
            if name.endswith(".tmp_end.ckpt"):
                continue
            else:
                select_name = name
        ckpt_path = os.path.join(ckpt_dir, select_name)
        print("Loading weights from checkpoints - {}".format(ckpt_path))
        state_dict = torch.load(ckpt_path, map_location=model.device)["state_dict"]
        for name, para in state_dict.items():
            if name == ['turn_decoder.weight', 'speaker_decoder.weight']:
                name = name.replace("decoder", "embedding")
                para = torch.cat([torch.zeros((1, para.shape[-1]), device=model.device), para], dim=0)
            new_state_dict[name] = para
    model.load_state_dict(new_state_dict, strict=False)
