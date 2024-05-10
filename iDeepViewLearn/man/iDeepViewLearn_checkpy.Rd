\name{iDeepViewLearn_checkpy}
\alias{iDeepViewLearn_checkpy}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
%%  ~~function to do ... ~~
Check python environment and packages.
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
Trains an iDeepViewLearn model with your choice of using default hyperparameter any time, using tuning dataset when both X_tune and y_tune are available, or using cross validation method y_train is available. Returns selected features, models trained, and classifier that goes into testing, and reconstructed data.
}
\usage{
iDeepViewLearn_checkpy(python_path)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{python_path}{A string to python enviroment. See Reticulate package.}
}
\details{
%%  ~~ If necessary, more details than the description above ~~
}
\value{
%%  ~Describe the value returned
The function should not return anything. If there is any error, please check your python enviornment and the packages in the python environment.
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
iDeepViewLearn_checkpy(python_path="~/.conda/envs/myenv/bin")
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
