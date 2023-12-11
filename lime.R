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
library(tidyverse)

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
df <- read.csv("IBM.csv")
df <- df %>% 
  mutate(Education = as.factor(case_when(Education == 1 ~ "Below College",
                                         Education == 2 ~ "College",
                                         Education == 3 ~ "Bachelor",
                                         Education == 4 ~ "Master",
                                         TRUE ~ "Doctor")),
         Age = as.factor(case_when(Age <= 25 ~ "Young",
                                  Age <= 54 ~ "Middle Aged",
                                   TRUE ~ "Senior")),
         BusinessTravel = as.factor(BusinessTravel),
         EducationField = as.factor(EducationField),
         Gender = as.factor(Gender),
         JobRole = as.factor(JobRole),
         MaritalStatus = as.factor(MaritalStatus),
         MonthlyIncome = as.factor(if_else(MonthlyIncome < median(MonthlyIncome), "Below Average", "Above Average")),
         JobLevel = as.factor(JobLevel),
         OverTime = as.factor(OverTime),
         Over18 = as.factor(Over18),
         Department = as.factor(Department),
         StockOptionLevel = as.factor(StockOptionLevel),
         Attrition = as.factor(case_when(Attrition == "No" ~ 0, Attrition == "Yes" ~ 1))) %>% 
  select(-c(EnvironmentSatisfaction, JobSatisfaction, PerformanceRating, EmployeeNumber,
            WorkLifeBalance, JobInvolvement, RelationshipSatisfaction, EmployeeCount, Over18))

############
# Data visualization
############

############
# Train/test split
############
set.seed(1)
train <- caret::createDataPartition(df$Attrition, p = 0.80, list = FALSE)
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
rf.cv <- train(Attrition ~ ., data = train.data,
               method = "rf",
               trControl = ctrl,                     
               tuneLenght = 5)
stopCluster(cl)
predict_rf <- predict(rf.cv, test.data)
print("Random Forest")
confusionMatrix(predict_rf, test.data$Attrition)


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
