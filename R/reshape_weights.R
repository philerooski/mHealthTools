np <- reticulate::import("numpy")

WEIGHTS_PATH <- "~/Desktop/mpower/yfg_weights"

get_weights <- function(p) {
  weights <- np$load(p, encoding='latin1')
  weights <- purrr::map(weights, function(arr) {
    if(length(dim(arr)) == 3) {
      arr <- reshape2::melt(arr) %>% 
        reshape2::acast(Var3 ~ Var2 ~ Var1)
    }
    return(arr)
  })
  return(weights)
}

main <- function() {
  weight_set_paths <- list.files(WEIGHTS_PATH, full.names = T,
                                 recursive = F, include.dirs = F,
                                 pattern = "fold.*")
  purrr::map(weight_set_paths, function(p) {
    list_index <- 1
    reshaped_weights <- vector("list", 50)
    weights <- get_weights(p)
    for (i in 1:length(weights)) {
      arr <- weights[[i]]
      reshaped_weights[[list_index]] <- arr
      list_index <- list_index + 1
      if(length(dim(arr)) == 3) { # is a conv layer
        reshaped_weights[[list_index]] <- array(0, dim(arr)[3])
        list_index <- list_index + 1
      }
    }
    saveRDS(reshaped_weights, file = file.path(
      dirname(p), "reshaped", paste0(basename(p), "_reshaped.RData")))
  })
}

combine_weights <- function(p) {
  weights_list <- list()
  weight_paths <- list.files(p, full.names=T)
  counter <- 1
  for (i in 1:length(weight_paths)) {
    path <- weight_paths[[i]]
    weights <- readRDS(path)
    print(length(weights))
    weights_list[[counter]] <- weights
    counter <- counter + 1
  }
  saveRDS(weights_list, file.path(p, "yfg_nn_weights.Rdata"))
}

#main()