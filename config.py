configuration = {

    # csrl data path
    "csrl_train_data_path": "./data/csrl_data/csrl_train.txt",
    "csrl_dev_data_path": "./data/csrl_data/csrl_dev.txt",
    "csrl_label_vocab_path": "./data/csrl_data/label.txt",
    "few_shot_csrl_en_data": "./data/csrl_data/csrl_en_15_shot.txt",
    "10_shot_csrl_data": "./data/csrl_data/fs-10percent/csrl_train.txt",
    "30_shot_csrl_data": "./data/csrl_data/fs-30percent/csrl_train.txt",
    "50_shot_csrl_data": "./data/csrl_data/fs-50percent/csrl_train.txt",
    "70_shot_csrl_data": "./data/csrl_data/fs-70percent/csrl_train.txt",

    # token-level and dialogue-level pretraining data
    "dialogue_train_path": ["./data/dialog_data/duconv.train.txt", "./data/dialog_data/persona.train.txt",
                            "./data/dialog_data/cmu_dog.train.txt"],
    # "dialogue_train_path": ["./data/dialog_data/duconv.debug.txt"],
    "dialogue_dev_path": ["./data/dialog_data/duconv.dev.txt", "./data/dialog_data/persona.dev.txt",
                          "./data/dialog_data/cmu_dog.dev.txt"],
    # "dialogue_dev_path": ["./data/dialog_data/duconv.dev.txt"],
    "lm_alignment_train_path": "./data/iwslt-2014/alignment.train.hpsi",
    "lm_alignment_dev_path": "./data/iwslt-2014/alignment.dev.hpsi",
    "lm_monolingual_train_path": ["./data/iwslt-2014/monolingual.en.train", "./data/iwslt-2014/monolingual.zh.train"],
    "lm_monolingual_dev_path": ["./data/iwslt-2014/monolingual.en.dev", "./data/iwslt-2014/monolingual.zh.dev"],
    # "srl_train_path": ["./data/srl_data/en.debug.srl"],
    "srl_train_path": ["./data/srl_data/en.train.srl", "./data/srl_data/zh.train.srl"],
    "srl_dev_path": ["./data/srl_data/en.dev.srl", "./data/srl_data/zh.dev.srl"]
}
