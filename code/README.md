The requirements.txt file was included in the repository so that others can easily replicate the findings by installing the same package versions that I used. To recreate my environment use the following command in shell: 
**pip install -r requirements.txt**.

The structure of this repository:
* data_preprocessing.ipynb - preprocessing raw data.
* exploratory_data_analysis.ipynb - exploratory data analysis on preprocessed data.
* modeling.ipynb - constructing an econometric model, verifying assumptions, and assessing goodness of fit.
* utils.py - a collection of small Python functions used for statistical inference and modeling.