iDeepViewLearn_train <- function(python_path, myseed=1234L,
                                 X_train, y_train=NULL, X_tune=NULL, y_tune=NULL,
                                 search_times=0L, best_comb=NULL, impt=NULL,
                                 top_rate=0.1, edged=NULL, vWeightd=NULL,
                                 epochs=1000L, fold=5L,
                                 plot=FALSE, verbose=TRUE, gpu=FALSE,
                                 normalization=TRUE){
  # prepare python
  library(reticulate)
  use_python(python_path)
  py_config()
  if (!py_available()) {
    stop("python not available")
  }
  #print(getwd())
  source_python("iDeepViewLearn.py")

  # train
  train_result = train(X_train=X_train, y_train=y_train, X_tune=X_tune, y_tune=y_tune,
                  comb_num=search_times, best_comb=best_comb, important=impt,
                  top_rate=0.1, edged=edged, vWeightd=vWeightd,
                  epochs=epochs, fold=fold, plot=plot, verbose=verbose, gpu=gpu,
                  normalization=normalization)
  return(train_result)
}
