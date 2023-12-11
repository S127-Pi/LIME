library(caret)
library(lime)
library(plyr)
library(randomForest)
library(doParallel)

df <- read.csv("Customer_Churn.csv")

############
# Data Pre-Processing
############
df$Complains <- as.factor(df$Complains)
df$Tariff.Plan <- as.factor(df$Tariff.Plan )
df$Churn <- as.factor(df$Churn)
df$Status <- as.factor(df$Status)
df$Age <- as.factor(df$Age)
df$Charge..Amount <- as.factor(df$Charge..Amount)

############
# Undersampling
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
# LIME
############
explainer <- lime(train.data, model = rf.cv)
explanation <- explain(test.data[1:5, ], explainer, labels = "0", n_features = 10)
plot_features(explanation)
