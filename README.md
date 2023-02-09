<br />
  <h1 align="center">Learning Admission-level Multimodal Patient Representations from EHRs
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
git clone https://github.com/bokajgd/hci-divalgo.git
cd hci-divalgo
```

### Virtual environment

Create and activate a new virtual environment your preferred way, and install the required packages in the requirements file.
Using pip, it is done by running

```bash
python3 -m venv divalgo
source divalgo/bin/activate
pip install -r requirements.txt
```

<!-- REPOSITORY STRUCTURE -->
## Repository structure

This repository has the following structure:

```
├── .streamlit             <- folder with app setup configuration file
├── divalgo                <- main folder with class and functions                      
│   ├── .streamlit         <- folder with app setup configuration file
│   ├── logos              <- logo and symbols for pages
|   |   └── ...
│   ├── pages              <- folder containing subpages for the streamlit app
|   |   └── ...
│   ├── demo.ipynb         <- jupyter notebook demonstrating the use of the class
│   ├── divalgo_class.py   <- script with class and main functions 
│   ├── utils.py           <- script with helper-functions for the class and app 
│   └── ☌frontpage.py      <- main streamlit file and frontpage
├── data                   <- folder containing the data - dogs vs wolf from Kaggle for the demonstration     
|   ├── dogs               <- folder containing images of dogs
|   └── wolves             <- folder containing images of wolves
├── .gitignore                 
├── README.md              <- the top-level README
└── requirements.txt       <- required packages
```


## Biliography

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

