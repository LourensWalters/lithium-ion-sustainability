Battery Cycle Life Prediction
=============================

This project is focused on the prediction of battery cycle life. The concept derives from a recent paper (published 
in Nature) called [Data-driven prediction of battery cycle life before capacity degradation](https://www.nature.com/articles/s41560-019-0356-8) by Kristen A. Severson et. al. 

The code for the data extraction component of the Nature article (not included in the Medium article code) is available 
here: 

* https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation

For this project we used the code from the previous repository but refactored it to be more easily navigable i.e. we 
created Object Oriented code from the modularised code. The code was also updated in that several changes have been 
made to Tensorflow since the code was written. We updated the code to be compatible with the latest 
stable Tensorflow  release. The code was developed using the PyCharm IDE, using a virtual environment and should 
therefore work if used with the included requirements.txt file (contains most recent versions of all libraries).

A write-up for this project can be found on the following blog: 

[Can Machine Learning make Lithium-ion Battery manufacturing more sustainable?](https://lourenswalters.github.io/2023/02/21/capstone-project-report.html)

The project code is organised as follows (based on CookieCutter data science template - see footnote): 


Project Organisation
--------------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

--------

Please feel free to check out the code, run it and improve it as you see fit. 

**To Note: Before running the code you will need to download the data.**

* You can download the data here: 
  [Data](<https://data.matr.io/1/>)
* Unzip the data into the "data" directory above (unzipped 9GB).
* You should now be ready to run the code. 

Also, you can use the _virtualenv_ in venv to run the code (it is up to date): 

_. .\venv\Scripts\activate_

Else you can use the _requirements.txt_ file to set up an environment of your own. 

There are various Jupyter notebooks in the notebook directory, that visualise results and can be used to run all of 
the data transformation and model building manually. 

Please let me know if you struggle to get the code work, I will try my best to assist you to get it up and running. 

I did install CUDA on my system, and do have an NVIDIA card. You can run the code without this, and you can also 
adapt it to run on GoogleCloud API, but I have not done so yet. Feel free to implement this though! Let me know if 
you need help on this. 

## References

* Attia, P. M., Grover, A., Jin, N., Severson, K. A., Markov, T. M., Liao, Y. H., Chen, M. H., Cheong, B., Perkins, N.
, Yang, Z., Herring, P. K., Aykol, M., Harris, S. J., Braatz, R. D., Ermon, S., & Chueh, W. C. (2020). Closed-loop optimization of fast-charging protocols for batteries with machine learning. Nature, 578(7795), 397–402. https://doi.org/10.1038/s41586-020-1994-5
* Kleppmann, M. (2017). Designing Data-Intensive Applications: The Big Ideas behind Reliable, Scalable, and 
  Maintainable Systems. In O’Reilly Media, Inc. https://www.oreilly.com/library/view/designing-data-intensive-applications/9781491903063/%0Ahttp://shop.oreilly.com/product/0636920032175.do
* Rattenbury, T., Hellerstein, J. M., Heer, J., Kandel, S., & Carreras, C. (2017). Principles of Data Wrangling.
* Severson, K. A., Attia, P. M., Jin, N., Perkins, N., Jiang, B., Yang, Z., Chen, M. H., Aykol, M., Herring, P. K., 
  Fraggedakis, D., Bazant, M. Z., Harris, S. J., Chueh, W. C., & Braatz, R. D. (2019). Data-driven prediction of battery cycle life before capacity degradation. Nature Energy, 4(5), 383–391. https://doi.org/10.1038/s41560-019-0356-8


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
