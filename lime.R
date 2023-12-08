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
library(gbm)
library(caret)
library(lime)
library(plyr)
library(randomForest)

############
# Data fetching
############

# Get dataset
odata <- odt(id = 4534)
# Access the actual data
df <- odata$data

############
# Preprocessing
############
df <- na.omit(df)
# Encoding target variable
df$Result <- mapvalues(df$Result , from = c(-1, 1), to = c(0, 1), warn_missing = TRUE)

############
# Data visualization
############

############
# Train/test split
############
set.seed(1)
train <- caret::createDataPartition(df$Result, p = 0.70, list = FALSE)
train.data <- df[train,]
test.data <- df[-train,]

############
# Model training/testing
############
# Logistic Regression
lg <- glm(Result ~ ., data = train.data, family = binomial)
pred <- predict(lg, test.data)
predict_reg <- factor(ifelse(pred >0.5, 1, 0))
#table(test.data$Result, predict_reg)
print("Logistic Regression")
confusionMatrix(predict_reg, test.data$Result)

# Random Forests
rf <- randomForest(Result~., data=train.data, proximity=TRUE)
predict_rf <- predict(rf, test.data)
#table(test.data$Result, predict_rf)

print("Random Forest")
confusionMatrix(predict_rf, test.data$Result)


############
# LIME
############

