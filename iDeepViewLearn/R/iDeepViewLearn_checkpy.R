iDeepViewLearn_checkpy <- function(python_path){
  library(reticulate)
  use_python(python_path)
  py_config()
  if (!py_available()) {
    stop("python not available")
  }
  # Check python packages
  
  print("The following packages used in the functions shoule be pre-installed with python:")
  print("random, itertools, os, copy.")
  print("Please check python installation if the above packages are not found.")
  
  if (!py_module_available("numpy")) {
    print("Installing numpy")
    py_install("numpy")
  }
  
  if (!py_module_available("torch")) {
    print("Installing torch")
    py_install("torch", pip = TRUE, repos = "https://conda.anaconda.org/conda-forge/")
  }
  
  print("If you plan to use GPU, please make sure the CUDA version of your GPU and your PyTorch package match")
#   print("The CUDA version of the current PyTorch version is:")
#   py_run_string("
# import torch
# print(torch.version.cuda)
# ")
  
  if (!py_module_available("sklearn")) {
    print("Installing sklearn")
    py_install("scikit-learn", pip = TRUE, repos = "https://pypi.org/simple/")
  }

  if (!py_module_available("sksurv")) {
    print("Installing sksurv")
    py_install("scikit-survival", pip = TRUE, repos = "https://pypi.org/simple/")
  }
  
  # if (!py_module_available("matplotlib")) {
  #   print("Installing matplotlib")
  #   py_install("matplotlib")
  # }
  
  # if (!py_module_available(package_name)) {
  #   py_install(package_name)
  # }

}





