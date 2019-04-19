# nlp-impl
Compilation of all the basic concepts of natural language processing such as text pre-processing, text classification, LSI &amp; LDA Modelling using SVD from the GENSIM python library.

## Important Instructions before runnin the project
* The data set used in the project is 'Consumer Complaints' from [this link](https://catalog.data.gov/dataset/consumer-complaint-database). In the project, since the dataset was too big a pickle file format of the same dataset has been used to avoid space and data read issues from the csv file.
* However, if you want to implement the algorithm on the csv file as is, just make sure to make the changes in read_csv function of pandas to create the dataframe from.

## Structure of the Project:
* All the techniques implemented in the project have been compiled into [functions.py](./functions.py).
* The implementation code has been written into [main.py](./main.py)

## Techniques and Algorithms Explained
* Text Pre-processing (Stemming, Lemmatization and stop words removal using nltk corpus and generating custom stop words basis our data)
  Fuction Name: **preprocess_**
* Text Classification Using Count & TF-IDF Vectorizer.
  Function Name(s): **text_classifier_, vectorizer_ & run_model_**
* Topic Modelling with LSI. (One using singular value decomposition algorithm & one from inbuilt class in GENSIM library for NLP).
  Function Name(s): **lsi_model_svd_ & lsi_model_**
* Topic Modelling with LDA (Latent Dirichlet Allocation - GENSIM Library)
  Function Name: **lda_model_**
