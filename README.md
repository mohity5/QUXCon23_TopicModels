## Topic Models: A tool for uncovering hidden themes in data.
Companion repository for the demo presented in the session **Topic Models: A tool for uncovering hidden themes in data** at **[Quant UX Con 23](https://www.quantuxcon.org).**


### Directory Structure

+ [main.md](main.md) : Markdown notebook.
+ [main.rmd](main.rmd) : R code.

**Python Modules**
+ [pyBase.ipynb](pyBase.ipynb) : Jupyter notebook for data loading and lemmatising.
+ [Modules/dataLoader.py](./Modules/dataLoader.py) : Loading data via scikit. **(20 newsgroup data)**
+ [Modules/lemmatiser.py](./Modules/lemmatiser.py) : Lemmatising docs.

**R Modules**
+ [Modules/R/rTokenising.r](./Modules/R/rTokenising.r) : Creating tokens
+ [Modules/R/zeroRows.r](./Modules/R/zeroRows.r) : Removing zero rows after preprocessing
+ [Modules/R/CreateJsonObj.r](./Modules/R/CreateJsonObj.r) : Creating JSON object for visualisation.

**rbase.rmd** is now in [Archive folder](./Archive/)