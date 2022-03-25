This project is forked from [CDialGPT](https://github.com/thu-coai/CDial-GPT).

# Introduction
This project included the source code of the TriKE-DialGPT proposed in our WSDM2022 paper.
 
# Instructions

## Environment
Please first prepare the environment and download the PLM checkpoints following the  [CDialGPT](https://github.com/thu-coai/CDial-GPT).

## Vocabulary
Our code needs to modify the vocabulary file.

Taking the checkpoint `CDial-GPT_LCCC-base` as our example.

1. First, make a copy `CDial-GPT_LCCC-base-MSKE`.
2. Then, replace the `CDial-GPT_LCCC-base-MSKE/vocab.txt` with the provided `mske_resources/CDial-GPT_LCCC-base-MSKE/vocab.txt`.
3. The differences between the two vocals lie in the last N lines, please check them.

## Dataset

Please download from this   [Google Drive](https://drive.google.com/file/d/19nBAbqRMjRv_Qf4gd1mvXS6mw690O1HO/view?usp=sharing)                                  


## Training
```shell script
python -u train.py --model_checkpoint CHECK_POINT_PATH --scheduler linear --pretrained --train_batch_size 8 --valid_steps 125000 --data_path DATA_PATH --gradient_accumulation_steps 4 --num_workers 0 --dataset_cache hke_verify_gpt_base  --knowledge_cache hke_verify >> hke_verify_gpt_base2.txt
```

The trained models will be saved at `runs/TIME_YOUR_COMPUTER_NAME`.

## Inference

Assuming `runs\May20_15-54-49_DESKTOP-MP3C892 ` is the model path, there will be a lot of checkpoints `*.pth`. If you want to infer dialogues based on `example.pth`， just rename `example.pth` to `pytorch_model.bin`  and then execute the following script:

```shell script
python infer.py  --model_checkpoint runs\May20_15-54-49_DESKTOP-MP3C892 --data_path  DATA_PATH --dataset_cache hke_verify_gpt_base  --knowledge_cache hke_verify  --out_path output.txt  --no_sample &
```

To select the final model, please use the validation losses in the log.

# Citation
```shell script
@inproceedings{DBLP:conf/wsdm/WuW0ZW22,
  author    = {Sixing Wu and
               Minghui Wang and
               Ying Li and
               Dawei Zhang and
               Zhonghai Wu},
  editor    = {K. Selcuk Candan and
               Huan Liu and
               Leman Akoglu and
               Xin Luna Dong and
               Jiliang Tang},
  title     = {Improving the Applicability of Knowledge-Enhanced Dialogue Generation
               Systems by Using Heterogeneous Knowledge from Multiple Sources},
  booktitle = {{WSDM} '22: The Fifteenth {ACM} International Conference on Web Search
               and Data Mining, Virtual Event / Tempe, AZ, USA, February 21 - 25,
               2022},
  pages     = {1149--1157},
  publisher = {{ACM}},
  year      = {2022},
  url       = {https://doi.org/10.1145/3488560.3498393},
  doi       = {10.1145/3488560.3498393},
  timestamp = {Fri, 18 Feb 2022 16:24:18 +0100},
  biburl    = {https://dblp.org/rec/conf/wsdm/WuW0ZW22.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```