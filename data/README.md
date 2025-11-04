---
license: cc-by-nc-sa-4.0
task_categories:
- text-classification
language:
- en
pretty_name: MetaHate
size_categories:
- 1M<n<10M
extra_gated_prompt: "Terms of Use for MetaHate."
extra_gated_fields:
  Name: text
  Surname: text
  Institution or organization: text
  Address: text
  Country: country
  Telephone: text
  Email address:  text
  How do you plan to use this dataset?: text
  I confirm I will use this dataset for academic and research purposes and I acknowledge that using this dataset for commercial purposes is prohibited: checkbox
  I confirm that I will not use this dataset to conduct any activity that causes harm to human subjects: checkbox
  I confirm that I will not try to identify the individuals whose texts are included in this dataset: checkbox
  I confirm that I will not share or redistribute this dataset with others: checkbox
  I confirm that if I violate any of these points access to the data will be revoked: checkbox
  I confirm that I will cite the reference below if I use this dataset: checkbox
  I understand that this agreement gives access to the publicly available part of MetaHate and the remaining instances require extra consent that is described in MetaHate_full_ToS: checkbox
---
# MetaHate: A Dataset for Unifying Efforts on Hate Speech Detection
This is MetaHate: a meta-collection of 36 hate speech datasets from social media comments.

## Dataset Structure
The dataset contains 1,226,202 social media posts in a TSV file. Each element contains the following fields:


| Field Name | Type | Possible Values | Description                                                          |
|------------|------|-----------------|----------------------------------------------------------------------|
| text       | str  | any             | Social media post. Each post is unique.            |
| label      | int  | 0, 1            | Label of the post. 0 for non-hate speech posts, 1 for hate speech.   |

## Usage
In order to use MetaHate you need to agree to our Terms and Conditions. Access to the complete meta-collection (1,226,202) will be granted only upon the submission of all relevant agreements for the derived datasets. Otherwise, we will only provide the access to the publicly available datasets (1,101,165 instances).

To access the full data, we require the original Terms of Use of the following works:

- [A Large Labeled Corpus for Online Harassment Research (Golbeck et al. 2017)](https://doi.org/10.1145/3091478.3091509)
- [The 'Call me sexist but' Dataset (Samory et al. 2021)](https://search.gesis.org/research_data/SDN-10.7802-2251)
- [Are You a Racist or Am I Seeing Things? Annotator Influence on Hate Speech Detection on Twitter (Waseem 2016)](https://doi.org/10.18653/v1/W16-5618)
- [Hateful Symbols or Hateful People? Predictive Features for Hate Speech Detection on Twitter (Waseem and Hovy 2016)](https://doi.org/10.18653/v1/N16-2013)
- [Aggression-annotated Corpus of Hindi-English Code-mixed Data (Kumar et al. 2018)](https://aclanthology.org/L18-1226)
- [#MeTooMA: Multi-Aspect Annotations of Tweets Related to the MeToo Movement (Gautam et al. 2020)](https://doi.org/10.1609/icwsm.v14i1.7292)
- [Pinpointing Fine-Grained Relationships between Hateful Tweets and Replies (Albanyan and Blanco 2022)](https://doi.org/10.1609/aaai.v36i10.21284)
- [Large-Scale Hate Speech Detection with Cross-Domain Transfer (Toraman, Şahinuç, and Yilmaz 2022)](https://aclanthology.org/2022.lrec-1.238)
- [Developing a Multilingual Annotated Corpus of Misogyny and Aggression (Bhattacharya et al. 2020)](https://aclanthology.org/2020.trac-1.25)

Send these agreements to paloma.piot@udc.es to access the full data.

## Disclaimer
This dataset includes content that may contain hate speech, offensive language, or other forms of inappropriate and objectionable material. The content present in the dataset is not created or endorsed by the authors or contributors of this project. It is collected from various sources and does not necessarily reflect the views or opinions of the project maintainers.

The purpose of using this dataset is for research, analysis, or educational purposes only. The authors do not endorse or promote any harmful, discriminatory, or offensive behaviour conveyed in the dataset.

Users are advised to exercise caution and sensitivity when interacting with or interpreting the dataset. If you choose to use the dataset, it is recommended to handle the content responsibly and in compliance with ethical guidelines and applicable laws.

The project maintainers disclaim any responsibility for the content within the dataset and cannot be held liable for how it is used or interpreted by others.

## Citation

If you use this dataset, please cite the following reference:

```bibtex
@article{Piot_Martín-Rodilla_Parapar_2024,
  title={MetaHate: A Dataset for Unifying Efforts on Hate Speech Detection},
  volume={18},
  url={https://ojs.aaai.org/index.php/ICWSM/article/view/31445},
  DOI={10.1609/icwsm.v18i1.31445},
  abstractNote={Hate speech represents a pervasive and detrimental form of online discourse, often manifested through an array of slurs, from hateful tweets to defamatory posts. As such speech proliferates, it connects people globally and poses significant social, psychological, and occasionally physical threats to targeted individuals and communities. Current computational linguistic approaches for tackling this phenomenon rely on labelled social media datasets for training. For unifying efforts, our study advances in the critical need for a comprehensive meta-collection, advocating for an extensive dataset to help counteract this problem effectively. We scrutinized over 60 datasets, selectively integrating those pertinent into MetaHate. This paper offers a detailed examination of existing collections, highlighting their strengths and limitations. Our findings contribute to a deeper understanding of the existing datasets, paving the way for training more robust and adaptable models. These enhanced models are essential for effectively combating the dynamic and complex nature of hate speech in the digital realm.},
  number={1},
  journal={Proceedings of the International AAAI Conference on Web and Social Media},
  author={Piot, Paloma and Martín-Rodilla, Patricia and Parapar, Javier},
  year={2024},
  month={May},
  pages={2025-2039}
}
```

## Acknowledgements
The authors thank the funding from the Horizon Europe research and innovation programme under the Marie Skłodowska-Curie Grant Agreement No. 101073351. The authors also thank the financial support supplied by the Consellería de Cultura, Educación, Formación Profesional e Universidades (accreditation 2019-2022 ED431G/01, ED431B 2022/33) and the European Regional Development Fund, which acknowledges the CITIC Research Center in ICT of the University of A Coruña as a Research Center of the Galician University System and the project PID2022-137061OB-C21 (Ministerio de Ciencia e Innovación, Agencia Estatal de Investigación, Proyectos de Generación de Conocimiento; supported by the European Regional Development Fund). The authors also thank the funding of project PLEC2021-007662 (MCIN/AEI/10.13039/501100011033, Ministerio de Ciencia e Innovación, Agencia Estatal de Investigación, Plan de Recuperación, Transformación y Resiliencia, Unión Europea-Next Generation EU).