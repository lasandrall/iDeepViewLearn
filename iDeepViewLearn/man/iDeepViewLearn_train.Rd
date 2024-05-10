\name{iDeepViewLearn_train}
\alias{iDeepViewLearn_train}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
%%  ~~function to do ... ~~
Trains an iDeepViewLearn model for multi-view data.
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
Trains an iDeepViewLearn model with your choice of using default hyperparameter any time, using tuning dataset when both X_tune and y_tune are available, or using cross validation method y_train is available. Returns selected features, models trained, and classifier that goes into testing, and reconstructed data.
}
\usage{
iDeepViewLearn_train(python_path, myseed=1234L,
                     X_train, y_train=NULL, X_tune=NULL, y_tune=NULL,
                     search_times=0L, best_comb=NULL, impt=NULL, outtype=NULL,
                     top_rate=0.1, edged=NULL, vWeightd=NULL,
                     epochs=1000L, fold=5L,
                     plot=FALSE, verbose=TRUE, gpu=FALSE,
                     normalization=TRUE)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{python_path}{A string to python enviroment. See Reticulate package.}

  \item{myseed}{An integer to set the seed. Need to append a letter L to the integer, for example, 1234L.}

  \item{X_train}{A list of d elements, where d is the number of views, being the training data. Each element is a view with dimension \eqn{n^d \times p^d}, where observations are on the rows and features are on the columns. \eqn{n^d} must be the same across all views, denote \eqn{n^d=n}, while \eqn{p^d} can be different.}

  \item{y_train}{\strong{If outcome variable is categorical, an integer vector of length \eqn{n^d} representing the classes; if continuous, a numeric vector of length \eqn{n^d}; if survival, a \eqn{n^d} by 2 matrix with the first column being event indicator (0=censored, 1=event) and the second column being survival time. This argument can be NULL. }}

  \item{X_tune}{A list of d elements, where d is the number of views, being the tuning data. This argument can be NULL. Each element is a view with dimension \eqn{n_{tune}^d \times p^d}, where observations are on the rows and features are on the columns. \eqn{n_{tune}^d} must be the same across all views. \eqn{p^d} can be different in each view, but must be consistent with the \eqn{p^d} in traning data.}

  \item{y_tune}{See y_train for format, with length of the vector or number of rows of the matrix \eqn{n_{tune}^d}. This argument can be NULL.}

  \item{search_times}{An integer of searching times for the hyperparameters. Need to append a letter L to the integer. To use the default hyperparameters, put 0L. To search once, put 1L. To search multiple times, for example, if you want to search 10 times, you may put 10L. In that case, we randomly selected 10 combinations from all possible grid search combinations. Increase the search times for an increased number of combinations to be searched, but the runtime could be longer. The maximum possible search times is the maximum grid search combinations, which is 192. }

  \item{best_comb}{A list of 4 elements being the best combination of hyperparameters, \strong{being K, the number of columns of the traied shared low dimensional latent code; the learning rate of the neural network for loss function; the learning rate of the neural net work for Z; the classifier parameter, C for SVM classifier and regressor, alpha for Survival SVM, and 1 for BIC (which does not mean anything and is only a placeholder); the panelty}. This argument can be NULL. If NULL, the algorithm will choose parameters according to the entry of search_times.}

  \item{impt}{An integer vector of length d, the number of views, being the number of features in the low-dimensional reconstruction for the d views. Need to append a letter L to the integer. For example, if we want top 50 features for the first view and top 25 features for the second view, put c(50L, 25L). This argument can be NULL. If NULL, the algorithm will use top_rate to decide.}
  
  \item{outtype}{\strong{A required argument to indicate the output type, including "categorical", "continuous", "survival", or "nothing" if you do not have outcome. }}

  \item{top_rate}{A number between 0 to 1 to indicate the top fraction of features being selected. If impt is available, this argument will not be used.}

  \item{edged}{A list of length d, the number of views, with each element being a dataframe representing the edge information of the variables to use the Laplacian version of iDeepViewLearn. The columns represents the edge connection. For example, in a row, if the first column of writes 1 and the second columns writes 26, then it means that there is an edge connecting variable 1 and variable 26. The number of rows, \eqn{r_d}, is the total amount of edges. This argument can be NULL. If NULL, the algorithm will carry out the standard iDeepViewLearn method.}

  \item{vWeighted}{A list of length d, the number of views, with each element being a vector representing the weight information of edges to use the Laplacian version of iDeepViewLearn. Each vector has length \eqn{r_d}, being the total amount of edges of view d. This argument can be NULL. If NULL, the algorithm will carry out the standard iDeepViewLearn method.}

  \item{epochs}{An integer indicating the epochs of traning. Need to append a letter L to the integer.}

  \item{fold}{An integer k indicating k-fold cross validation, if cross validation is used. Need to append a letter L to the integer.}

  \item{plot}{TRUE or FALSE to plot line charts of unsupervised 1, unsupervised 2, and Z loss history.}

  \item{verbose}{TRUE or FALSE to output training status and best combination of hyperparameters during training.}

  \item{gpu}{TRUE or FALSE to use gpu.}
  \item{normalization}{TRUE or FALSE to normalize the training and tuning data. We recommend using TRUE to prevent predictors with large norms to shadow the result.}
}
\details{
%%  ~~ If necessary, more details than the description above ~~
If both X_tune and y_tune are available, the algorithm will by default using the validation dataset. If X_tune is available while y_tune is not, or if neither is available, if specified a search_time other than 0L, the algorithm will automatically do cross validation.
}
\value{
%%  ~Describe the value returned
The function will return train_result, a list of 8 elements. To see the elements, use double square brackets. See below for more detail.
%%  If it is a LIST, use
%%  \item{comp1 }{Description of 'comp1'}
%%  \item{comp2 }{Description of 'comp2'}
%% ...
The following arguments are needed if you want to proceed with testing.
\item{selected_features}{A list of d elements, where d is the number of views. Can be obtained by train_result[[1]]. Each element is a list of the indices of selected features that go into the low-dimensional reconstruction. The selected features of the d-th view can be obtained by train_result[[1]][[d]].}
\item{model}{A list object being the trained models. Can be obtained by train_result[[2]].}
\item{clf}{A list object being the trained classifier if y_train is available. \strong{Returns SVM classifier for categorical data, SVM Regressor for continuous data, Survival SVM estimator for survival data, and the loss of the last iteration for missing outcome types.} Can be obtained by train_result[[3]].}
\item{best_comb}{A list of 4 element being the hyperparameters used in the model. Can be obtained by train_result[[4]].}
The following arguments provide latent code and the low-dimensional representations.
\item{Ztrain}{A matrix of \eqn{n \times K}, where n is the number of observations and K is one of the hyperparameters of the model, being the shared latent code of the original data containing all features. Can be obtained by train_result[[5]].}
\item{GZ}{A list of d elements, where each element has dimension \eqn{n\times p^d}, being the reconstructed data in feature selection. Can be obtained by train_result[[6]].}
\item{Zprime}{A matrix of \eqn{n \times K}, being the shared latent code using the selected features only. Can be obtained by train_result[[7]].}
\item{RZprime}{A list of d elements, where each element has dimention \eqn{n \times p_{impt}^d}, being the nonlinear approximations. Use in downstream analyses. Can be obtained by train_result[[8]].}
}

\references{
Hengkang Wang, Han Lu, Ju Sun, Sandra E. Safo, \emph{Interpretable Deep Learning Methods for Multiview Learning}, submitted.
}
\author{
%%  ~~who you are~~
}
\note{
%%  ~~further notes~~
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
\code{\link{iDeepViewLearn_data_example},\link{iDeepViewLearn_test}}
}

\examples{
######## import library and data example
library(iDeepViewLearn)
data("iDeepViewLearn_data_example")

######## Train default iDeepViewLearn
# Use default hyperparameters (no tuning data needed) to train an iDeepViewLearn model.
# By ground truth of the data example, the first 50 variables are truly important.
# Here we chose to normalize data, i.e. normalize X_train and X_tune for each view.
train_result = iDeepViewLearn_train(python_path="~/.conda/envs/myenv/bin", myseed=1234L,
                                    X_train = X_train, y_train = y_train,
                                    X_tune=NULL, y_tune=NULL, search_times=0L, # use default
                                    best_comb=NULL, impt=c(50L,50L),top_rate=0.1,
                                    edged=NULL, vWeightd=NULL, epochs=1000L,
                                    fold=5L, plot=FALSE, verbose=TRUE, gpu=FALSE,
                                    normalization=TRUE)
######## Train iDeepViewLearn with tuning
# Use tuning dataset, search 5 times
train_result_2 = iDeepViewLearn_train(python_path="~/.conda/envs/myenv/bin", myseed=1234L,
                                      X_train = X_train, y_train = y_train,
                                      X_tune=X_tune, y_tune=y_tune, # use tuning dataset
                                      search_times=5L, best_comb=NULL, # search 5 times
                                      impt=c(50L, 50L), top_rate=0.1, 
                                      edged=NULL, vWeightd=NULL, 
                                      epochs=1000L, fold=5L,
                                      plot=FALSE, verbose=TRUE, gpu=FALSE,
                                      normalization = TRUE)
# Use cross-validation, search 5 times
train_result_3 = iDeepViewLearn_train(python_path="~/.conda/envs/myenv/bin", myseed=1234L,
                                      X_train = X_train, y_train = y_train,
                                      X_tune=NULL, y_tune=NULL, # no tuning data, use CV
                                      search_times=5L, best_comb=NULL, # search 5 times
                                      impt=c(50L, 50L), top_rate=0.1, 
                                      edged=NULL, vWeightd=NULL, 
                                      epochs=1000L, fold=5L,
                                      plot=FALSE, verbose=TRUE, gpu=FALSE,
                                      normalization = TRUE)


######## Obtaining training result
# To get the indices of selected features for each view
selected_1 = train_result[[1]][[1]]
selected_2 = train_result[[1]][[2]]
# To get True Positive Rate and False Positive Rate for the first view
# The ground truth (gt) is that the first 50 variables are important
# and the 51~500 variables are not important.
# Note: python indices start with 0
gt_imp_1 = 0:49
gt_not_imp_1 = 50:499
all_var_1 = 0:499
unselected_1 = all_var_1[!(all_var_1 %in% selected_1)]
TP_1 = sum(selected_1 %in% gt_imp_1)
TN_1 = sum(unselected_1 %in% gt_not_imp_1)
FP_1 = sum(selected_1 %in% gt_not_imp_1)
FN_1 = sum(unselected_1 %in% gt_imp_1)
TPR_1 = TP_1 / (TP_1 + FN_1) * 100
FPR_1 = FP_1 / (FP_1 + TN_1) * 100
F_1 = TP_1 / (TP_1 + 0.5 * (FP_1 + FN_1)) * 100
# To get training accuracy
# Pass X_train and y_train to iDeepViewLearn_test() and obtain the accuracy
test_result = iDeepViewLearn_test(python_path = "~/.conda/envs/myenv/bin",
                                  X_train = X_train, X_test = X_train,
                                  y_train = y_train, y_test = y_train,
                                  features = train_result[[1]],
                                  model = train_result[[2]],
                                  clf = train_result[[3]],
                                  best_comb = train_result[[4]],
                                  normalization = TRUE)
train_acc = test_result[[2]]
# To obtain G(Z), Z', and R(Z')
GZ = train_result[[6]]
Zprime = train_result[[7]]
RZprime = train_result[[8]]

######## Import Laplacian data example
library(iDeepViewLearn)
data("iDeepViewLearn_Laplacian_data_example")

######## Train iDeepViewLearn-Laplacian
# Use validation dataset and default hyperparameters to train an iDeepViewLearn model.
# By ground truth of the data example, 21 variables are truly important. See gt.
# Here we chose to NOT normalize data.
train_result_L = iDeepViewLearn_train(python_path="~/.conda/envs/myenv/bin",
                                      myseed=1234L,
                                      X_train = X_train, y_train = y_train,
                                      X_tune=X_tune, y_tune=y_tune,
                                      search_times=0L, best_comb = NULL,
                                      impt=c(21L,21L), top_rate=0.1,
                                      edged=edge_ls, vWeightd=weight_vec_ls,
                                      epochs=1000L, fold=NULL,
                                      plot=FALSE, verbose=TRUE, gpu=FALSE,
                                      normalization = TRUE)
# The followed analysis should be similar.
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory (show via RShowDoc("KEYWORDS")):
% \keyword{ ~kwd1 }
% \keyword{ ~kwd2 }
% Use only one keyword per line.
% For non-standard keywords, use \concept instead of \keyword:
% \concept{ ~cpt1 }
% \concept{ ~cpt2 }
% Use only one concept per line.
