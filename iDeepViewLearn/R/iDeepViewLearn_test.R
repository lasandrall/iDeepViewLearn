iDeepViewLearn_test = function(python_path,
                               X_train, X_test, best_comb, features, model, clf,
                               y_train=NULL, y_test=NULL,
                               top_rate=0.1, edged=NULL, vWeightd=NULL, epochs=1000L,
                               plot=FALSE, verbose=TRUE, gpu=FALSE,
                               normalization=TRUE){
  # prepare python
  library(reticulate)
  use_python(python_path)
  py_config()
  if (!py_available()) {
    stop("python not available")
  }
  source_python("iDeepViewLearn.py")
  # test
  test_result = test(X_train=X_train, y_train=y_train, X_tune=X_test, y_tune=y_test,
                     comb=best_comb, features=features, model_2=model, clf=clf,
                     top_rate=top_rate, edged=edged, vWeightd=vWeightd, epochs=epochs,
                     plot=plot, verbose=verbose, gpu=gpu, normalization=normalization)
}
