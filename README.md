## Fast Gradient Projection Method (FGPM)
This repository contains code to reproduce results from the paper:

[**Adversarial Training with Fast Gradient Projection Method against Synonym Substitution based Text Attacks**](https://arxiv.org/abs/2008.03709) (AAAI 2021) <br />
Xiaosen Wang, Yichen Yang, Yihe Deng, Kun He <br />

## Datesets
There are three datasets used in our experiments. Download and place the file `train.csv` and `test.csv` of the three datasets under the directory `/data/ag_news`, `/data/dbpedia` and `/data/yahoo_answers`, respectively.

- [AG's News](https://s3.amazonaws.com/fast-ai-nlp/ag_news_csv.tgz)
- [DBPedia](https://s3.amazonaws.com/fast-ai-nlp/dbpedia_csv.tgz)
- [Yahoo! Answers](https://s3.amazonaws.com/fast-ai-nlp/yahoo_answers_csv.tgz)

## Dependencies
There are two dependencies for this project. Download and put `glove.840B.300d.txt` and `counter-fitted-vectors.txt` to the directory `/data/`.

- [GloVe vecors](http://nlp.stanford.edu/data/glove.840B.300d.zip)
- [Counter fitted vectors](https://github.com/nmrksic/counter-fitting/blob/master/word_vectors/counter-fitted-vectors.txt.zip)

## Requirements
- python 3.6.5
- numpy 1.15.2
- tensorflow-gpu 1.12.0
- keras 2.2.0

## File Description

- `textcnn.py`,`textrnn.py`,`textbirnn.py` : The models for CNN, LSTM and Bi-LSTM.
- `train.py`: Normally or adversarially training models.
- `utils.py` : Helper functions for building dictionaries, loading data, or processing embedding matrix etc.
- `build_embeddings.py` : Generating the dictionary, embedding matrix and distance matrix.
- `FGPM.py` : Fast Grandient Projection Method.
- `attack.py`: Attack models with FGPM.
- `Config.py`: Settings of datasets, models and attacks.

## Experiments

1. Generating the dictionary, embedding matrix and distance matrix:

    ```shell
    python build_embeddings.py --data ag_news --data_dir ./data/
    ```
    
    You could use our pregenerated data by downloading and place [aux_files](https://drive.google.com/file/d/18jGoR4mj4iSlO9SPV6LNPBvH3KhBo397/view?usp=sharing) into the directory `/data/`.

2. Training the models normally:

    ```shell
    python train.py  --data ag_news -nn_type textcnn --train_type org --num_epochs=2 --num_checkpoints=2 --data_dir ./data/ --model_dir ./model/
    ```
    (You will get a directory named like `1583313019_ag_news_org` in path `/model/runs_textcnn`)
    
    You could also use our trained model by downloading and placing [runs_textcnn](https://drive.google.com/file/d/1772TeKlR3iBTTEkpSr3S2FO4eONrlT0T/view?usp=sharing), [runs_textrnn](https://drive.google.com/file/d/1mz50SLpxjnw2BOOwifXFytJesndQlgSg/view?usp=sharing) and [runs_textbirnn](https://drive.google.com/file/d/1yUDEbndLBLbawkdZctS3m9I-Ov5zjRkf/view?usp=sharing) into the directory `/model/`.

3. Attack the normally trained model by FGPM:

    ```shell
    python attack.py --nn_type textcnn --data ag_news --train_type org --time 1583313019 --step 2 --grad_upd_interval=1 --max_iter=30  --data_dir ./data/ --model_dir ./model/
    ```
    (**Note that you may get another timestamp, check the file name of the model in `/model/runs_textcnn`**)

4. Training the models by ATFL to enhance robustness:

    ```shell
    python train.py  --data ag_news -nn_type textcnn --train_type adv --num_epochs=10 --num_checkpoints=10 --grad_upd_interval=1 --max_iter=30 --data_dir ./data/ --model_dir ./model/
    ```
    (You will get a directory named like `1583313121_ag_news_adv` in path `/model/runs_textcnn`)

    You could also use our trained model by downloading and placing [runs_textcnn](https://drive.google.com/file/d/1772TeKlR3iBTTEkpSr3S2FO4eONrlT0T/view?usp=sharing), [runs_textrnn](https://drive.google.com/file/d/1mz50SLpxjnw2BOOwifXFytJesndQlgSg/view?usp=sharing) and [runs_textbirnn](https://drive.google.com/file/d/1yUDEbndLBLbawkdZctS3m9I-Ov5zjRkf/view?usp=sharing) into the directory `/model/`.

5. Attack the adversarially trained model by FGPM:

    ```shell
    python attack.py --nn_type textcnn --data ag_news --train_type adv --time 1583313121 --step 3 --grad_upd_interval=1 --save=True --max_iter=30 --data_dir ./data/ --model_dir ./model/
    ```
    (**Note that you may get another timestamp, check the file name of the model in `/model/runs_textcnn`**)

## More details

Most experiments setting have been provided in our paper. Here we provide some more details to help reproduce our results.

+ For normal training, we set `num_epochs` to `2` on CNN models and `3` on RNN models. For adversarial training, we train 10 epochs for all models except for RNN models of Yahoo! Answers dataset with `3` epochs.

+ The parameter `max_iter` denotes the maximum number of iterations, namely `N` in the FGPM algorithm. According to the average length of the samples, we empirically set `max_iter` to `30` on `ag_news`, `40` on `dbpedia`, and `50` on `yahoo_answers`. Moreover, to speed up the training by ATFL on Yahoo! Answers, we calculate the gradient every `5` optimal synonym substitution operations (i.e. `grad_upd_interval = 5`) .

+ In order to maintain the fairness for comparison, we restrict the candidate words in the first 4 clostest synonyms of each word. While implement adversarial trainging, to obtain more adversarial examples, we do not have such restriction.

+ In order to improve the readability of adversarial examples, we have enabled stop words by default to prohibit the attack algorithm from replacing words such as `the/a/an` with synonyms. Stop words can be seen in `Config.py`. You can also turn off stop words by setting `stop_words = False` when attack or adversarial training.

## Contact

Questions and suggestions can be sent to xswanghuster@gmail.com.

## Citation

If you find this code and data useful, please consider citing the original work by authors:

```
@article{wang2021Adversarial,
  title={Adversarial Training with Fast Gradient Projection Method against Synonym Substitution based Text Attacks},
  author={Xiaosen Wang and Yichen Yang and Yihe Deng and Kun He},
  journal={AAAI Conference on Artificial Intelligence},
  year={2021}
}
```
