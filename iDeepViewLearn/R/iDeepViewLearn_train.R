iDeepViewLearn_train <- function(python_path, myseed=1234L,
                                 X_train, y_train=NULL, X_tune=NULL, y_tune=NULL,
                                 search_times=0L, best_comb=NULL, impt=NULL, outtype = NULL,
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
  
  My_Package_Name_path <- system.file("python", package = "iDeepViewLearn")
  print(My_Package_Name_path)
  reticulate::py_run_string(paste0("import sys; sys.path.append('", My_Package_Name_path, "')"))
  
  #reticulate::source_python("iDeepViewLearn.py")
  reticulate::source_python(system.file("python",
                                        "iDeepViewLearn.py",
                                        package = "iDeepViewLearn",
                                        mustWork = TRUE))
  
  #source_python("iDeepViewLearn.py")

  # train
  train_result = train(X_train=X_train, y_train=y_train, X_tune=X_tune, y_tune=y_tune,
                  comb_num=search_times, best_comb=best_comb, important=impt, outtype=outtype,
                  top_rate=top_rate, edged=edged, vWeightd=vWeightd,
                  epochs=epochs, fold=fold, plot=plot, verbose=verbose, gpu=gpu,
                  normalization=normalization, myseed=myseed)
  return(train_result)
}
