import json
from utils import calc_f1
import argparse


def scores(in_file, return_res=False):
    gold_all_srl_list = []
    pred_all_srl_list = []

    gold_inter_srl_list = []
    pred_inter_srl_list = []

    gold_inner_srl_list = []
    pred_inner_srl_list = []

    with open(in_file, "r", encoding="utf-8") as fr:
        for line in fr:
            jo = json.loads(line.strip())
            gold_all_srl_list.append(jo["gold_span"])
            pred_all_srl_list.append(jo["pred_span"])
            gold_inter_srl_list.append(jo["gold_inter_span"])
            pred_inter_srl_list.append(jo["pred_inter_span"])
            gold_inner_srl_list.append(jo["gold_inner_span"])
            pred_inner_srl_list.append(jo["pred_inner_span"])

    overall_f1 = calc_f1(gold_all_srl_list, pred_all_srl_list)
    inter_f1 = calc_f1(gold_inter_srl_list, pred_inter_srl_list)
    inner_f1 = calc_f1(gold_inner_srl_list, pred_inner_srl_list)
    if return_res:
        return inner_f1['F'], inter_f1['F'], overall_f1['F']
    else:
        print("results on all args: {}".format(overall_f1))
        print("results on inter args: {}".format(inter_f1))
        print("results on inner args: {}".format(inner_f1))


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--model_name", type=str)
    arg_parser.add_argument("--dataset_name", choices=["duconv", "persona", "dog"], default="persona")
    args = arg_parser.parse_args()

    test_path_dict = {
        "duconv": "csrl_test.txt",
        "persona": "csrl_en_persona_test.txt",
        "dog": "csrl_en_dog_test.txt"
    }
    file_name = "{}.{}".format(args.model_name, test_path_dict[args.dataset_name])

    scores(file_name)
