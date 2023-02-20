\name{iDeepViewLearn_sim_downstream_outcome}
\alias{iDeepViewLearn_sim_downstream_outcome}
\docType{data}
\title{
%%   ~~ data name/kind ... ~~
Randomly simulated outcome data of different kind to demonstrate possible downstream analysis using shared or reconstructed low-dimensinal representations.
}
\description{
%%  ~~ A concise (1-5 lines) description of the dataset. ~~
}
\usage{data("iDeepViewLearn_sim_downstream_outcome")}
\format{
  An R object with 5 items, each of length 350. Note that this version of outcome columns are completed randomly generated and merged with the simulation dataset. These outcome columns are for demonstration purposes only.
  \describe{
    \item{binary_outcome}{A integer vector of length 350 obtaining values of 0 or 1. Used for example of downstream analyses with binary outcomes.}
    \item{categorical_outcome}{A integer vector of length 350 obtaning values 0, 1, 2, or 3. Used for example of downstream analyses with categorical outcomes.}
    \item{continuous_outcome}{A numeric vector of length 350. Used for example of downstream analyses with continuous outcomes.}
    \item{survival_time, survival_event}{Each being a vector of length 350. Used for example of downstream analyses with survival outcomes}
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
data("iDeepViewLearn_sim_downstream_outcome")
# See iDeepViewLearn_test for details.
}
\keyword{datasets}