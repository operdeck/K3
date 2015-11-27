library(data.table)
library(dplyr)

# https://www.kaggle.com/c/telstra-recruiting-network/data

train <- fread("data/train.csv")
test <- fread("data/test.csv")
log_feature <- fread("data/log_feature.csv")
event_type <- fread("data/event_type.csv")
resource_type <- fread("data/resource_type.csv")
severity_type <- fread("data/severity_type.csv")

# join all together - this will result in a much longer dataset

train_all <- left_join(train, resource_type) %>% 
  left_join(log_feature) %>%
  left_join(severity_type) %>%
  left_join(event_type)

test_all <- left_join(test, resource_type) %>% 
  left_join(log_feature) %>%
  left_join(severity_type) %>%
  left_join(event_type)

# there are test locations not present in the train set but many are
# id's do not overlap

length(unique(test$location))
length( intersect(unique(train$location), unique(test$location)) )

# predict 'fault_severity' (which is 0, 1 or 2)


