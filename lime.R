if (!requireNamespace("mlr3oml", quietly = TRUE)) {
  install.packages("mlr3oml")
}
if (!requireNamespace("mlr3", quietly = TRUE)) {
  install.packages("mlr3")
}
if (!requireNamespace("h2o", quietly = TRUE)) {
  install.packages("h2o")
}
if (!requireNamespace("caret", quietly = TRUE)) {
  install.packages("caret")
}
if (!requireNamespace("lime", quietly = TRUE)) {
  install.packages("lime")
}

library(mlr3oml)
library(mlr3)
library(h2o)
library(caret)
library(lime)

############
# Data fetching
############

# Get dataset
odata <- odt(id = 151)
# Access the actual data
df <- odata$data

############
# Data visualization
############

############
# Train/test split
############
df <- na.omit(df)
set.seed(1)
train <- caret::createDataPartition(df$Class, p = 0.70, list = FALSE)
train.data <- df[train,]
test.data <- df[-train,]

############
# Model training/testing
############


############
# LIME
############

