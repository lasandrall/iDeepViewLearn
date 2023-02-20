\name{iDeepViewLearn_data_example}
\alias{iDeepViewLearn_data_example}
\docType{data}
\title{
%%   ~~ data name/kind ... ~~
Data example for iDeepViewLearn.
}
\description{
%%  ~~ A concise (1-5 lines) description of the dataset. ~~
}
\usage{data("iDeepViewLearn_data_example")}
\format{
  An R object with 6 items. 2 views, each with 350 observations and 500 variables. For simplicity and for illustration purpose, both view have the same number of features in this example. Note that while the number of observations in each view must be the same, the number of features in each view can be different. 
  \describe{
    \item{X_train, X_tune, Xtest}{Each being a list of 2 views. Each with has 350 observations and 500 variables.}
    \item{y_train, y_tune, y_test}{Each being an integer vector of length 350, obtaining values 1 and 2 indicating the class.}
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
data("iDeepViewLearn_data_example")
# See iDeepViewLearn_train or iDeepViewLearn_test for details.
}
\keyword{datasets}
