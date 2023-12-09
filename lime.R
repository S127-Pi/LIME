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

if (!requireNamespace("randomForest", quietly = TRUE)) {
  install.packages("randomForest")
}

if (!requireNamespace("doParallel", quietly = TRUE)) {
  install.packages("doParallel")
}


library(mlr3oml)
library(mlr3)
library(h2o)
library(gbm)
library(caret)
library(lime)
library(plyr)
library(randomForest)
library(doParallel)

############
# Data fetching
############

# Get dataset
# id = 151 electricity
id = 4534 #phishing
odata <- odt(id = id)
# Access the actual data
df <- odata$data

############
# Preprocessing
############
df <- na.omit(df)
# Remapping target values
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
# lg <- glm(Result ~ ., data = train.data, family = binomial)
# pred <- predict(lg, test.data)
# predict_reg <- factor(ifelse(pred >0.5, 1, 0))
# print("Logistic Regression")
# confusionMatrix(predict_reg, test.data$Result)

# Random Forests
cl <- makePSOCKcluster(7)
registerDoParallel(cl)
ctrl <- trainControl(method = "cv",number = 10,
                     allowParallel = TRUE, 
                     verboseIter = FALSE)
rf.cv <- train(Result ~ ., data = train.data,
               method = "rf",
               trControl = ctrl,                     
               tuneLenght = 5)
stopCluster(cl)
predict_rf <- predict(rf.cv, test.data)
print("Random Forest")
confusionMatrix(predict_rf, test.data$Result)


############
# LIME
############
explainer <- lime(train.data, model = rf.cv)
explanation <- explain(test.data[20:25, ], explainer, n_labels = 1, n_features = 10)
plot_features(explanation)

# Tune LIME algorithm
explanation_tuned <- explain(
  x = test.data[20:25,],
  explainer = explainer,
  n_permutations = 5000,
  kernel_distance = "euclidean",
  kernel_width = 3,
  n_features = 10,
  n_labels = 1,
)
plot_features(explanation_tuned)
