\name{iDeepViewLearn_test}
\alias{iDeepViewLearn_test}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
%%  ~~function to do ... ~~
Tests performance and provides low-dimensinal representations given the trained iDeepViewLearn model.
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
Takes the trained iDeepViewLearn model and applies on the test dataset. Returns model used, classfier and accuracy if applied, and low-dimensional representation in testing stage.
}
\usage{
iDeepViewLearn_test(python_path,
                    X_train, X_test, best_comb, features, model, clf,
                    y_train=NULL, y_test=NULL,
                    top_rate=0.1, edged=NULL, vWeightd=NULL, epochs=1000L,
                    plot=FALSE, verbose=TRUE, gpu=FALSE, normalization=TRUE)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{python_path}{A string to python enviroment. See Reticulate package.}

  \item{X_train}{A list of d elements, where d is the number of views, being the training data. Each element is a view with dimension \eqn{n^d \times p^d}, where observations are on the rows and features are on the columns. \eqn{n^d} must be the same across all views, denote \eqn{n^d=n}, while \eqn{p^d} can be different.}
  
  \item{X_test}{A list of d elements, where d is the number of views, being the testing data.  Each element is a view with dimension \eqn{n_{test}^d \times p^d}, where observations are on the rows and features are on the columns. \eqn{n_{test}^d} must be the same across all views. \eqn{p^d} can be different in each view, but must be consistent with the \eqn{p^d} in traning data.}

  \item{best_comb}{A list of 5 elements being the best combination of hyperparameters obtained from the training result.}

  \item{features}{A list of d elements, each being a list of each view's indices of the selected features obtained from the training result.}

  \item{model}{A list object obtained from the training result, being the model trained.}

  \item{clf}{A list obtained from the training result, being the classifier trained.}

  \item{y_train}{An integer vector of length \eqn{n^d} representing the classes. This argument can be NULL.}

  \item{y_test}{An integer vector of length \eqn{n_{test}^d} representing the classes. This argument can be NULL.}

  \item{top_rate}{A number between 0 to 1 to indicate the top fraction of features being selected. If impt is available, this argument will not be used.}

  \item{edged}{A list of length d, the number of views, with each element being a dataframe representing the edge information of the variables to use the Laplacian version of iDeepViewLearn. The columns represents the edge connection. For example, in a row, if the first column of writes 1 and the second columns writes 26, then it means that there is an edge connecting variable 1 and variable 26. The number of rows, \eqn{r_d}, is the total amount of edges. This argument can be NULL. If NULL, the algorithm will carry out the standard iDeepViewLearn method.}

  \item{vWeighted}{A list of length d, the number of views, with each element being a vector representing the weight information of edges to use the Laplacian version of iDeepViewLearn. Each vector has length \eqn{r_d}, being the total amount of edges of view d. This argument can be NULL. If NULL, the algorithm will carry out the standard iDeepViewLearn method.}

  \item{epochs}{An integer indicating the epochs of traning. Need to append a letter L to the integer.}

  \item{plot}{TRUE or FALSE to plot line charts of unsupervised 1, unsupervised 2, and Z loss history.}

  \item{verbose}{TRUE or FALSE to output training status and best combination of hyperparameters during training.}

  \item{gpu}{TRUE or FALSE to use gpu.}
  \item{normalization}{TRUE or FALSE to normalize the testing data. We recommend using TRUE to prevent predictors with large norms to shadow the result.}
}
\details{
%%  ~~ If necessary, more details than the description above ~~
}
\value{
%%  ~Describe the value returned
The function will return train_result, a list of 5 elements. To see the elements, use double square brackets. See below for more detail.
%%  If it is a LIST, use
%%  \item{comp1 }{Description of 'comp1'}
%%  \item{comp2 }{Description of 'comp2'}
%% ...
The following arguments are related to model and model assessment.
\item{pred}{A list of integers being the predicted class. Takes NULL if clf is NULL or if y_test is not available. Can be obtained by test_result[[1]].}
\item{acc}{A number of classification accuracy. Takes NULL if clf is NULL or if y_test is not available. Can be obtained by test_result[[2]].}
The following arguments are low-dimensinal representations during the testing stage.
\item{Zprimetest}{A matrix of \eqn{n_{test} \times K}, being the shared latent code using the selected features only during the testing stage. Can be obtained by test_result[[3]].
}
\item{RZprimetest}{A list of d elements, where each element has dimention \eqn{n_{test} \times p_{impt}^d}, being the nonlinear approximations. Use in downstream analyses. Can be obtained by test_result[[4]].}
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
\code{\link{iDeepViewLearn_data_example},\link{iDeepViewLearn_train}}
}
\examples{
######## import library and data example
library(iDeepViewLearn)
data("iDeepViewLearn_data_example")

######## Train standard iDeepViewLearn
# Use validation dataset and default hyperparameters to train an iDeepViewLearn model.
# By ground truth of the data example, the first 50 variables are truly important.
# Here we chose to normalize data, i.e. normalize X_train and X_tune for each view.
train_result = iDeepViewLearn_train(python_path="~/.conda/envs/myenv/bin", myseed=1234L,
                                    X_train = X_train, y_train = y_train,
                                    X_tune=X_tune, y_tune=y_tune, search_times=0L,
                                    best_comb=NULL, impt=c(50L,50L), top_rate=0.1,
                                    edged=NULL, vWeightd=NULL, epochs=1000L,
                                    fold=5L, plot=FALSE, verbose=TRUE, gpu=FALSE,
                                    normalization=TRUE)
######## Testing stage
test_result = iDeepViewLearn_test(python_path = "~/.conda/envs/myenv/bin",
                                  X_train = X_train, X_test = X_test,
                                  y_train = y_train, y_test = y_test,
                                  features = train_result[[1]],
                                  model = train_result[[2]],
                                  clf = train_result[[3]],
                                  best_comb = train_result[[4]],
                                  normalization=TRUE)

######## Obtaining testing result
# To get the testing classification accuracy
test_acc = test_result[[2]]
# To get the Z' and R(Z') during the testing stage
Zprimetest = test_result[[3]]
RZprimetest = test_result[[4]]

######## Downstream analyses with Z'
# To use the Z', the shared low-dimensional representation of selected features
# in the testing stage to conduct downstream analysis
# Example: real data analysis and figure 6 in the paper.
data("iDeepViewLearn_sim_downstream_outcome")
# Example: Simple Linear Regression
slr_data = as.data.frame(cbind(Zprimetest, continuous_outcome))
slr_mod = lm(continuous_outcome~., data = slr_data)
# Example: Logistic Regression
lr_data = as.data.frame(cbind(Zprimetest, binary_outcome))
lr_mod = glm(binary_outcome~., family = binomial, data = lr_data)
# Example: SVM with categorical outcome
library(e1071)
svm_data = as.data.frame(cbind(Zprimetest, categorical_outcome))
svm_fit = svm(categorical_outcome~., data = svm_data,
              kernel = "radial", cost = 1, sacle = FALSE)

######## Downstream analyses with R(Z')
# To use the R(Z'), the reconstructed low-dimensional representation of selected 
# features of each view in the testing stage to conduct downstream analysis
# Example: K-means clustering followed by Survival Analysis
cluster_1 = kmeans(RZprimetest[[1]], 3, algorithm = "Lloyd")
cluster_2 = kmeans(RZprimetest[[2]], 3, algorithm = "Lloyd")
library(survival)
surv_data = as.data.frame(cbind(cluster_1$cluster, cluster_2$cluster,
                                survival_time, survival_event))
surv_fit = survfit(Surv(survival_time, survival_event)~V1+V2, data = surv_data)
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
