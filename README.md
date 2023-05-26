<br />
  <h1 align="center">Multimodal Elementary Clinical Information Embeddings: Encoding structured, time series and free-text data from ICU electronic health records into a common algebraic space
 </h1>
 <h2 align="center">Master's Thesis, Cognitive Sceince @ Aarhus University 2023</h2>

  <p align="center">
    Jakob Grøhn Damgaard
    <br />
</p>

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About the project</a></li>
    <li><a href="#getting-started">Getting started</a></li>
    <li><a href="#repository-structure">Repository structure</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About the project


<!-- GETTING STARTED -->
## Getting started

### Cloning repository and creating virtual environment

To obtain the the code, clone the following repository.

```bash
git clone https://github.com/bokajgd/multimodal-representation-learning-ehr.git
cd multimodal-representation-learning-ehr
```

### Virtual environment

Create and activate a new virtual environment your preferred way, and install the required packages in the requirements file.
Using pip, it is done by running

```bash
python3 -m venv ehr
source ehr/bin/activate
pip install -r requirements.txt
```

<!-- REPOSITORY STRUCTURE -->
## Repository structure

This repository has the following structure:

```
.
├── README.md
├── data
│   ├── feature_sets
│   ├── mimic-iii-clinical-database-1.4
│   └── misc
├── outputs
│   ├── eval_outputs
│   └── model_outputs
├── poetry.lock
├── pyproject.toml
├── requirements.txt
── src
│   ├── __init__.py
│   ├── cohort
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   ├── create_cohort_with_prediction_times.py
│   │   └── utils.py
│   ├── config
│   │   ├── __init__.py
│   │   ├── data
│   │   ├── default_config.yaml
│   │   ├── eval
│   │   ├── model
│   │   ├── preprocessing
│   │   ├── project
│   │   ├── sweeper
│   │   └── train
│   ├── evaluation
│   │   ├── descriptive_stats.py
│   │   ├── misc_plots.py
│   │   └── tsne_plot.py
│   ├── features
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   ├── create_full_feature_set_and_save_to_disk.py
│   │   ├── expand_features_to_dichotomous.py
│   │   ├── feature_specification
│   │   ├── generate_flattened_features.py
│   │   ├── static_and_flattened_features
│   │   ├── text_features
│   │   └── utils
│   └── model_training
│       ├── __init__.py
│       ├── __pycache__
│       ├── basemodel.py
│       ├── col_name_inference.py
│       ├── conf_utils.py
│       ├── data_schema.py
│       ├── dataclasses_schemas.py
│       ├── full_config.py
│       ├── get_search_space.py
│       ├── model.py
│       ├── model_evaluator.py
│       ├── model_pipeline.py
│       ├── model_specs.py
│       ├── outputs
│       ├── preprocessing.py
│       ├── process_manager_setup.py
│       ├── project.py
│       ├── to_disk.py
│       ├── train.py
│       ├── train_full_model.py
│       ├── train_model_functions.py
│       ├── train_multiple_models.py
│       ├── trainer_spawner.py
│       └── utils.py
├── tests
└── text_models
```

## Contact
**Jakob Grøhn Damgaard** 
<br />
[bokajgd@gmail.com](mailto:bokajgd@gmail.com?subject=[GitHub]%20stedsans)


[<img align="left" alt="Jakob Grøhn Damgaard | Twitter" width="30px" src="https://cdn.jsdelivr.net/npm/simple-icons@v3/icons/twitter.svg" />][twitter2]
[<img align="left" alt="Jakob Grøhn Damgaard | LinkedIn" width="30px" src="https://cdn.jsdelivr.net/npm/simple-icons@v3/icons/linkedin.svg" />][linkedin2]

<br />

</details>

[twitter2]: https://twitter.com/JakobGroehn
[linkedin2]: https://www.linkedin.com/in/jakob-gr%C3%B8hn-damgaard-04ba51144/


 <br>
 
## License
Shield: [![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg

