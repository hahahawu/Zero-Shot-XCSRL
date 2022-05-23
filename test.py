import argparse
import torch
import tqdm
import copy
import os
import json

from processor import Processor
from dataset import load_CSRL_data
from models.xsrl_model import XSRLModel
from utils import fine_print_res, calc_f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_name", choices=["duconv", "persona", "dog"], default="duconv")
    parser.add_argument("--test_data_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--gpus", type=str, default='0')
    hparams = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = hparams.gpus
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ckpt_dir = "checkpoints" + os.path.sep + hparams.model_name + os.path.sep + "checkpoints" + os.path.sep\
               + hparams.model_name
    ckpt_name = os.listdir(ckpt_dir)
    select_name = ckpt_name[0]
    for name in ckpt_name:
        if name.endswith(".tmp_end.ckpt"):
            continue
        else:
            select_name = name
    ckpt_path = os.path.join(ckpt_dir, select_name)
    config_path = os.path.join("checkpoints", hparams.model_name, "h_params.json")
    config = json.load(open(config_path, "r", encoding="utf-8"))
    config = argparse.Namespace(**config)

    test_data_path = hparams.test_data_path
    if test_data_path is None:
        test_path_dict = {
            "duconv": "data/csrl_data/csrl_test.txt",
            "persona": "data/csrl_data/csrl_en_persona_test.txt",
            "dog": "data/csrl_data/csrl_en_dog_test.txt"
        }
        test_data_path = test_path_dict[hparams.dataset_name]

    test_data = load_CSRL_data(test_data_path)
    processor = Processor('csrl', pretrain_model_name=config.language_model, label_path=config.csrl_label_vocab_path)
    csrl_label_vocab = processor.label_vocab
    csrl_idx2label = dict(zip(csrl_label_vocab.values(), csrl_label_vocab.keys()))

    model = XSRLModel.load_from_checkpoint(checkpoint_path=ckpt_path, map_location=device, hparams=config)
    model.eval()

    gold_all_srl_list = []
    pred_all_srl_list = []

    gold_inter_srl_list = []
    pred_inter_srl_list = []

    gold_inner_srl_list = []
    pred_inner_srl_list = []

    if not os.path.exists(hparams.output_dir):
        os.mkdir(hparams.output_dir)

    out_file_name = "{}.{}".format(hparams.model_name, test_data_path.split("/")[-1])
    output_file_path = os.path.join(hparams.output_dir, out_file_name)

    with open(output_file_path, "w", encoding="utf-8") as fw:
        for i, example in tqdm.tqdm(enumerate(test_data)):
            sent_id = example["sent_id"]
            input_words = copy.deepcopy(example["words"])
            data_example = processor.encode_csrl_case(example)
            predictions = model.predict(data_example, csrl_idx2label, need_mapping=False)[0]
            gold_role_spans, gold_inner_role_spans, gold_inter_role_spans, _ = fine_print_res(input_words, example["label"])
            pred_role_spans, pred_inner_role_spans, pred_inter_role_spans, _ = fine_print_res(input_words, predictions)
            gold_all_srl_list.append(gold_role_spans)
            pred_all_srl_list.append(pred_role_spans)
            gold_inter_srl_list.append(gold_inter_role_spans)
            pred_inter_srl_list.append(pred_inter_role_spans)
            gold_inner_srl_list.append(gold_inner_role_spans)
            pred_inner_srl_list.append(pred_inner_role_spans)

            info = {"id": i, "sent_id": sent_id, "gold_span": gold_role_spans, "gold_inner_span": gold_inner_role_spans,
                    "gold_inter_span": gold_inter_role_spans, "pred_span": pred_role_spans,
                    "pred_inner_span": pred_inner_role_spans, "pred_inter_span": pred_inter_role_spans,
                    "sent": " ".join(input_words)}
            fw.write(json.dumps(info, ensure_ascii=False) + "\n")

    overall_f1 = calc_f1(gold_all_srl_list, pred_all_srl_list)
    inter_f1 = calc_f1(gold_inter_srl_list, pred_inter_srl_list)
    inner_f1 = calc_f1(gold_inner_srl_list, pred_inner_srl_list)
    print("results on all args: {}".format(overall_f1))
    print("results on inter args: {}".format(inter_f1))
    print("results on inner args: {}".format(inner_f1))
