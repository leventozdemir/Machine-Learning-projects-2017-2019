![pipeline_svm](https://user-images.githubusercontent.com/51120437/125528287-0634fba8-3e18-47ec-b9a6-062799c2303a.png)

# SVM Pipeline
## To understande this code you have to read the [SVM Repository](https://github.com/leventozdemir/SVM_Algorithm) .

### the most importan thing is tune the tfidf feature extractor in terms of:

- max_df: The maximal document frequency of a term to be allowed, in order to avoid common terms generally occurring in documents.
- max_features: Number of top features to consider; we have only used 8000 till now for experiment purposes
- sublinear_tf: Scaling term frequency with the logarithm function or not.
- smooth_idf: Adding an initial 1 to the document frequency or not, similar tothe smoothing for the term frequency.
- The grid search model searches for the optimal set of parameters throughout the entire pipeline.
