# yelp_abcs

## About This Project
Open data on restuarant health ratings served as the initial motivation for this project. We wanted to explore the possibility of supplementing restaurant health ratings with user generated reviews. In this proof of concept we were able to demonstrate how to acquire, clean, and merge disparate data sets, including from open data portals and Yelp’s academic dataset. Data availability proved to be a significant challenge in this project. The Yelp Academic Dataset provided full corpus of the written reviews data, however, the set is far from complete and only represents a small sample of the businesses in Toronto or Las Vegas. Several different models were tested including logistic regression, gradient boosted trees, gradient boosted trees with principal component analysis (PCA), and random forest classifier. These models did not produce highly accurate (accuracy between 0.75 to 0.76) or specific results (AUC between 0.5 and 0.6), these results are encouraging as a proof of concept for further exploration perhaps with a keen eye towards feature selection and dimensionality reduction. 


> [Public Health vs The People: Analysis to Compare Las Vegas & Toronto Restaurant Inspections with Diners’ Yelp Reviews](https://docs.google.com/document/d/180dsiT59t037TFR3RKjoXAGO5QnHr-wWGZTt22bg3wM/edit?usp=sharing)

## Folders
* Stag:
    * Folder for code that moves the downloaded raw data files into warehouse
    * Start script with stag_ and correspond to schema in warehouse

* Clean:
    * Folder for code that cleans/ processes the data into analysis ready dataframes
    * Start script with clean_ and correspond to schema in warehouse

* Explore:
    * Folder for notebooks with descriptives and viz
    * Eda_ notebooks

* Model:
    * Folder for code related to models to be built
    * This maybe notebooks and .py scripts


