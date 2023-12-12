# Install and load the iml package
if (!require(iml)) {
  install.packages("iml")
}
if (!require(counterfactuals)) {
  install.packages("counterfactuals")
}
if (!require(GGally)) {
  install.packages("GGally")
}
library(iml)
library(caret)
library(lime)
library(plyr)
library(randomForest)
library(doParallel)
library(rpart)
library(iml)
library(iml)
library(counterfactuals)
library(GGally)
library(ggplot2)

df <- read.csv("Customer_Churn.csv")
df$Churn <- mapvalues(df$Churn , from = c(0, 1), to = c("Non-Churn", "Churn"), warn_missing = TRUE)
df$Status <- mapvalues(df$Status , from = c(1, 2), to = c("active", "non-active"), warn_missing = TRUE)
############
# Data Pre-Processing
############
df$Complains <- as.factor(df$Complains)
df$Tariff.Plan <- as.factor(df$Tariff.Plan )
df$Churn <- as.factor(df$Churn)
df$Churn <- relevel(df$Churn, ref = "Non-Churn")
df$Status <- as.factor(df$Status)
df$Status <- relevel(df$Status, ref = "active")
df$Age <- as.factor(df$Age)
df$Charge..Amount <- as.factor(df$Charge..Amount)

############
# Data Visualization
############
ggplot(data = df, aes(x = Churn, fill = Churn)) +
  geom_bar() +
  stat_count()


############
# Downsampling
############

# df_down <- downSample(x = df[, -which(names(df) == "Churn")], 
#                                 y = df$Churn)
# colnames(df_down)[length(df_down)] <- "Churn"
# 
# set.seed(1)
# df_down <- df_down[sample(nrow(df_down)), ]
# 
# df <- df_down

############
# Train/test split
############
set.seed(1)
train <- caret::createDataPartition(df$Churn, p = 0.70, list = FALSE)
train.data <- df[train,]

#Down sampling training data
train.data <- downSample(x = train.data[, -which(names(train.data) == "Churn")], 
                                 y = train.data$Churn)
colnames(train.data)[length(train.data)] <- "Churn"


test.data <- df[-train,]

############
# Model training/testing
############

# Random Forests
cl <- makePSOCKcluster(7)
registerDoParallel(cl)
ctrl <- trainControl(method = "cv",number = 10,
                     allowParallel = TRUE, 
                     verboseIter = FALSE)
rf.cv <- train(Churn ~ ., data = train.data,
               method = "rf",
               trControl = ctrl,                     
               tuneLenght = 10)
stopCluster(cl)
predict_rf <- predict(rf.cv, test.data)
print("Random Forest")
confusionMatrix(predict_rf, test.data$Churn)

############
# Counterfactual
############
predictor = Predictor$new(rf.cv, type = "prob")
x_interest = test.data[2, ]
predictor$predict(x_interest) 

wi_classif = WhatIfClassif$new(predictor, n_counterfactuals = 5)
cfactuals = wi_classif$find_counterfactuals(
  x_interest, desired_class = "Churn", desired_prob = c(0.5, 1)
)
cfactuals$data
cfactuals$evaluate()
cfactuals$plot_freq_of_feature_changes()
cfactuals$plot_parallel()

############
# LIME
############
explainer <- lime(train.data, model = rf.cv)
explanation <- explain(test.data[1:5, !(names(test.data) %in% "Churn")], explainer, labels = "Churn", n_features = 5)
plot_features(explanation)


############
# Tuned LIME
############
tuned_explanation <- explain(test.data[1:5, !(names(test.data) %in% "Churn")], explainer, labels = "Churn", 
                             n_permutations = 500,
                             dist_fun = "euclidean",
                             kernel_width = 3,
                             feature_select = "highest_weights",
                             n_features = 5)
plot_features(tuned_explanation)

############
# Shapley Values
############
# Create a Predictor object
predictor <- Predictor$new(rf.cv, data = train.data, y = "Churn")

# Compute the Shapley values
shapley <- Shapley$new(predictor, x.interest = test.data[1:5, ])
shapley_values <- shapley$plot()

# Print the Shapley values
print(shapley_values)
