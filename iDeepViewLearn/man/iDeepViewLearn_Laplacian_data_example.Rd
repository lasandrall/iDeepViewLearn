\name{iDeepViewLearn_Laplacian_data_example}
\alias{iDeepViewLearn_Laplacian_data_example}
\docType{data}
\title{
%%   ~~ data name/kind ... ~~
Data with prior information example for iDeepViewLearn with Laplacian.
}
\description{
%%  ~~ A concise (1-5 lines) description of the dataset. ~~
}
\usage{data("iDeepViewLearn_Laplacian_data_example")}
\format{
  An R object with 9 items. 2 views, each with 350 observations and 500 variables. For simplicity and for illustration purpose, both view have the same number of features and the same edge and weight information in this example. Note that while the number of observations in each view must be the same, the number of features in each view can be different. Also note that even if the edge and the weight information of different view can be different. 
  \describe{
    \item{X_train, X_tune, Xtest}{Each being a list of 2 views. Each with has 350 observations and 500 variables.}
    \item{y_train, y_tune, y_test}{Each being an integer vector of length 350, obtaining values 1 and 2 indicating the class.}
  }
  The first 50 variables are connected with a graph. See Figure 4 Left for a visualization for this example data. The ground truth is that 21  features is truly important in data generating.
  \describe{
    \item{gt}{A vector of length 21 being the indices of the ground truth of important features in data generating. Note that for python indices start with 0.}
  }
  Prior information.
  \describe{
    \item{edge_ls}{A list of 2 dataframes with dimension \eqn{49\times 2} being the edge information of 2 views. The columns represents the edge connection. For example, in a row, if the first column of writes 1 and the second columns writes 26, then it means that there is an edge connecting variable 1 and variable 26. The number of rows, 49 in this example, is the total number of edges.}
    \item{weight_vec_ls}{A list of 2 vectors with length 49 being the weight information of 2 views. }
  }
}
\details{
%%  ~~ If necessary, more details than the __description__ above ~~
}
\source{
%%  ~~ reference to a publication or URL from which the data were obtained ~~
}
\references{
Hengkang Wang, Han Lu, Ju Sun, Sandra E. Safo, \emph{Interpretable Deep Learning Methods for Multiview Learning}, submitted.
}
\examples{
data("iDeepViewLearn_Laplacian_data_example")
# See iDeepViewLearn_train for details.
}
\keyword{datasets}
