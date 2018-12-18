library(tidyverse)
library(synapser)

PREDICTIONS_SYN_ID = "syn17013591"
YFG_SUBMISSION_SYN_ID = "syn10942871"

read_syn_csv <- function(syn_id) {
  f <- synGet(syn_id)
  df <- read_csv(f$path)
  return(df)
}

prep_predictions <- function(predictions) {
  predictions %>%
    group_by(healthCode, assay, weight_set) %>% 
    summarize(prediction = mean(value)) %>% 
    ungroup()
}

clean_yfg_submission <- function(sub) {
  feature_cols <- purrr::map(1:20, ~ paste0("Feature", as.character(.))) %>% unlist()
  test_table <- synTableQuery("select * from syn10733842")$asDataFrame() %>% 
    as_tibble() %>% 
    select(healthCode, recordId) %>% 
    left_join(sub, by = "recordId") %>% 
    select(healthCode, dplyr::one_of(!!feature_cols))
  return(test_table)
}

make_comparison <- function(predictions, sub) {
  predictions %>%
    left_join(sub, by = "healthCode") %>%
    distinct()
}

main <- function() {
  syn <- synLogin()
  predictions <- read_syn_csv(PREDICTIONS_SYN_ID)
  predictions <- prep_predictions(predictions)
  yfg_submission <- read_syn_csv(YFG_SUBMISSION_SYN_ID)
  yfg_submission <- clean_yfg_submission(yfg_submission)
  comparison <- make_comparison(predictions, yfg_submission)
}