---
task_categories:
- text-classification
language:
- en
tags:
- hate speech
size_categories:
- 100K<n<1M
extra_gated_prompt: "You agree to not use the dataset to conduct any activity that causes harm to human subjects."
extra_gated_fields:
  Please provide more information on why you need this dataset and how you plan to use it:
    type: text
---

# English Hate Speech Superset


This dataset is a superset (N=360,493) of posts annotated as hateful or not. It results from the preprocessing and merge of all available English hate speech datasets in April 2024. These datasets were identified through a systematic survey of hate speech datasets conducted in early 2024. We only kept datasets that:
- are documented
- are publicly available
- focus on hate speech, defined broadly as "any kind of communication in speech, writing or behavior, that attacks or uses pejorative or discriminatory language with reference to a person or a group on the basis of who they are, in other words, based on their religion, ethnicity, nationality, race, color, descent, gender or other identity factor" (UN, 2019)

The survey procedure is further detailed in [our survey paper](https://aclanthology.org/2024.woah-1.23/).

## Data access and intended use
Please send an access request detailing how you plan to use the data. The main purpose of this dataset is to train and evaluate hate speech detection models, as well as study hateful discourse online. This dataset is NOT intended to train generative LLMs to produce hateful content. 

## Columns

The dataset contains six columns:
- `text`: the annotated post
- `labels`: annotation of whether the post is hateful (`== 1`) or not (`==0`). As datasets have different annotation schemes, we systematically binarized the labels.
- `source`: origin of the data (e.g., Twitter)
- `dataset`: dataset the data is from (see "Datasets" part below)
- `nb_annotators`: number of annotators by post
- `post_author_country_location`: post author country location, when it could be inferred. Details on the inference in [our survey paper](https://aclanthology.org/2024.woah-1.23/).

## Datasets

The datasets that compose this superset are:
- Automated Hate Speech Detection and the Problem of Offensive Language (`davidson` in the `dataset` column)
  - [paper link](https://ojs.aaai.org/index.php/ICWSM/article/view/14955/14805)
  - [raw data link](https://github.com/t-davidson/hate-speech-and-offensive-language)
- When Does a Compliment Become Sexist? Analysis and Classification of Ambivalent Sexism Using Twitter Data (`compliment_sexist` in the `dataset` column)
  - [paper link](https://aclanthology.org/W17-2902/)
  - [raw data link](https://github.com/AkshitaJha/NLP_CSS_2017)
- Detecting Online Hate Speech Using Context Aware Models (`fox_news`)
  - [paper link](https://aclanthology.org/R17-1036/)
  - [raw data link](https://github.com/sjtuprog/fox-news-comments/blob/master/fox-news-comments.json)
- Hate Speech Dataset from a White Supremacy Forum (`white_supremacy`)
  - [paper link](https://www.aclweb.org/anthology/W18-5102.pdf)
  - [raw data link](https://github.com/Vicomtech/hate-speech-dataset)
- Peer to Peer Hate: Hate Speech Instigators and Their Targets (`melsherief`)
  - [paper link](https://ojs.aaai.org/index.php/ICWSM/article/view/15038/14888)
  - [raw data link](https://github.com/melsherief/hate_speech_icwsm18)
- Hate Lingo: A Target-based Linguistic Analysis of Hate Speech in Social Media (`melsherief`)
  - [paper link](https://ojs.aaai.org/index.php/ICWSM/article/view/15041)
  - [raw data link](https://github.com/melsherief/hate_speech_icwsm18)
- Anatomy of Online Hate: Developing a Taxonomy and Machine Learning Models for Identifying and Classifying Hate in Online News Media (`anatomy_online_hate`)
  - [paper link](https://ojs.aaai.org/index.php/ICWSM/article/view/15028/14878)
  - [raw data link](https://www.dropbox.com/s/21wtzy9arc5skr8/ICWSM18%20-%20SALMINEN%20ET%20AL.xlsx?dl=0)
- CONAN - COunter NArratives through Nichesourcing: a Multilingual Dataset of Responses to Fight Online Hate Speech (`CONAN`)
  - [paper link](https://www.aclweb.org/anthology/P19-1271.pdf)
  - [raw data link](https://github.com/marcoguerini/CONAN)
- A Benchmark Dataset for Learning to Intervene in Online Hate Speech (`benchmark`)
  - [paper link](https://aclanthology.org/D19-1482/)
  - [raw data link](https://github.com/jing-qian/A-Benchmark-Dataset-for-Learning-to-Intervene-in-Online-Hate-Speech)
- Multilingual and Multi-Aspect Hate Speech Analysis (`MLMA`)
  - [paper link](https://aclanthology.org/D19-1474/)
  - [raw data link](https://huggingface.co/datasets/nedjmaou/MLMA_hate_speech)
- Overview of the HASOC track at FIRE 2019: Hate Speech and Offensive Content Identification in Indo-European Languages (`hasoc`)
  - [paper link](https://dl.acm.org/doi/pdf/10.1145/3368567.3368584?download=true)
  - [raw data link](https://hasocfire.github.io/hasoc/2019/dataset.html)
- Constructing interval variables via faceted Rasch measurement and multitask deep learning: a hate speech application (`measuring-hate-speech`)
  - [paper link](https://arxiv.org/abs/2009.10277)
  - [raw data link](https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech)
- Detecting East Asian Prejudice on Social media (`east_asian`)
  - [paper link](https://aclanthology.org/2020.alw-1.19/)
  - [raw data link](https://zenodo.org/records/3816667)
- Learning from the Worst: Dynamically Generated Datasets to Improve Online Hate Detection (`learning from the worst`)
  - [paper link](https://aclanthology.org/2021.acl-long.132/)
  - [raw data link](https://raw.githubusercontent.com/bvidgen/Dynamically-Generated-Hate-Speech-Dataset/main/Dynamically%20Generated%20Hate%20Dataset%20v0.2.2.csv)
- HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection (`hatexplain`)
  - [paper link](https://arxiv.org/abs/2012.10289)
  - [raw data link](https://github.com/hate-alert/HateXplain/tree/master/Data)
- An Expert Annotated Dataset for the Detection of Online Misogyny (`online-misogyny`)
  - [paper link](https://aclanthology.org/2021.eacl-main.114/)
  - [raw data link](https://github.com/ellamguest/online-misogyny-eacl2021/blob/main/data/final_labels.csv)
- Introducing CAD: the Contextual Abuse Dataset (`CAD`)
  - [paper link](https://aclanthology.org/2021.naacl-main.182.pdf)
  - [raw data link](https://zenodo.org/records/4881008)
- Hatemoji: A Test Suite and Adversarially-Generated Dataset for Benchmarking and Detecting Emoji-based Hate (`hatemoji-build`)
  - [paper link](https://aclanthology.org/2022.naacl-main.97/)
  - [raw data link](https://github.com/HannahKirk/Hatemoji)
- The Gab Hate Corpus: A collection of 27k posts annotated for hate speech (`GHC`)
  - [paper link](https://link.springer.com/article/10.1007/s10579-021-09569-x)
  - [raw data link](https://osf.io/edua3/)
- ETHOS: an Online Hate Speech Detection Dataset (`ETHOS`)
  - [paper link](https://arxiv.org/pdf/2006.08328.pdf)
  - [raw data link](https://github.com/intelligence-csd-auth-gr/Ethos-Hate-Speech-Dataset)
- Analyzing the Proliferation of Hate Speech in Parler (`Parler`)
  - [paper link](https://aclanthology.org/2022.woah-1.11.pdf)
  - [raw data link](https://github.com/NasLabBgu/parler-hate-speech/blob/main/parler_annotated_data.csv)
- SemEval-2023 Task 10: Explainable Detection of Online Sexism (`EDOS`)
  - [paper link](https://aclanthology.org/2023.semeval-1.305/)
  - [raw data link](https://github.com/rewire-online/edos)
    
## Additional datasets on demand
In our survey, we identified six additional datasets that are not public but can be requested to the authors. The full list is:
- Hateful Symbols or Hateful People? Predictive Features for Hate Speech Detection on Twitter
  - [paper link](https://www.aclweb.org/anthology/N16-2013)
  - [request link here](https://github.com/ZeerakW/hatespeech)
- Are You a Racist or Am I Seeing Things? Annotator Influence on Hate Speech Detection on Twitter
  - [paper link](https://aclanthology.org/W16-5618/)
  - [request link here](https://github.com/ZeerakW/hatespeech)
- Large Scale Crowdsourcing and Characterization of Twitter Abusive Behavior
  - [paper link](https://arxiv.org/pdf/1802.00393.pdf)
  - [request link here](https://dataverse.mpi-sws.org/dataset.xhtml?persistentId=doi:10.5072/FK2/ZDTEMN)
- hatEval, SemEval-2019 Task 5: Multilingual Detection of Hate Speech Against Immigrants and Women in Twitter
  - [paper link](https://www.aclweb.org/anthology/S19-2007)
  - [request link here](https://hatespeechdata.com/competitions.codalab.org/competitions/19935)
- “Call me sexist, but...” : Revisiting Sexism Detection Using Psychological Scales and Adversarial Samples 
  - [paper link](https://ojs.aaai.org/index.php/ICWSM/article/view/18085/17888)
  - [request link here](https://doi.org/10.7802/2251)
- Large-Scale Hate Speech Detection with Cross-Domain Transfer
  - [paper link](https://aclanthology.org/2022.lrec-1.238/)
  - [request link here](https://github.com/avaapm/hatespeech)

## Preprocessing

We drop duplicates. In case of non-binary labels, the labels are binarized (hate speech or not). We replace all usernames and links by fixed tokens to maximize user privacy. Further details on preprocessing can be found in the preprocessing code [here](https://github.com/manueltonneau/hs_geographic_survey).


## Citation
Please cite our [survey paper](https://aclanthology.org/2024.woah-1.23/) if you use this dataset.

```bibtex
@inproceedings{tonneau-etal-2024-languages,
    title = "From Languages to Geographies: Towards Evaluating Cultural Bias in Hate Speech Datasets",
    author = {Tonneau, Manuel  and
      Liu, Diyi  and
      Fraiberger, Samuel  and
      Schroeder, Ralph  and
      Hale, Scott  and
      R{\"o}ttger, Paul},
    editor = {Chung, Yi-Ling  and
      Talat, Zeerak  and
      Nozza, Debora  and
      Plaza-del-Arco, Flor Miriam  and
      R{\"o}ttger, Paul  and
      Mostafazadeh Davani, Aida  and
      Calabrese, Agostina},
    booktitle = "Proceedings of the 8th Workshop on Online Abuse and Harms (WOAH 2024)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.woah-1.23",
    pages = "283--311",
    abstract = "Perceptions of hate can vary greatly across cultural contexts. Hate speech (HS) datasets, however, have traditionally been developed by language. This hides potential cultural biases, as one language may be spoken in different countries home to different cultures. In this work, we evaluate cultural bias in HS datasets by leveraging two interrelated cultural proxies: language and geography. We conduct a systematic survey of HS datasets in eight languages and confirm past findings on their English-language bias, but also show that this bias has been steadily decreasing in the past few years. For three geographically-widespread languages{---}English, Arabic and Spanish{---}we then leverage geographical metadata from tweets to approximate geo-cultural contexts by pairing language and country information. We find that HS datasets for these languages exhibit a strong geo-cultural bias, largely overrepresenting a handful of countries (e.g., US and UK for English) relative to their prominence in both the broader social media population and the general population speaking these languages. Based on these findings, we formulate recommendations for the creation of future HS datasets.",
}

```