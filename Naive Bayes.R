train_LP <- read.csv("C:/Users/Sandhya/OneDrive/Data Science Practical Learning/Projects/Loan Prediction/train_LP.csv")
train_LP$Data <- "Train"
test_LP <- read.csv("C:/Users/Sandhya/OneDrive/Data Science Practical Learning/Projects/Loan Prediction/test_LP.csv")
test_LP$Loan_Status <- NA
test_LP$Data <- "Test"
summary(train_LP)
summary(test_LP)

model.nb <- e1071::naiveBayes(Loan_Status~Gender+Married+Dependents+Education+Self_Employed+
                                Credit_History, data=train_LP)

train_LP$Pred <- predict(model.nb, train_LP[-c(13)], type="raw")
conf_Mat <- table(train_LP$Loan_Status ,train_LP$Pred )

train_LP$Pred <- ifelse(train_LP$Pred=="Y",1,0)
library(pROC)
auc(train_LP$Loan_Status,train_LP$Pred)

# Cross Validation
install.packages("klaR")
install.packages("caret")

library("klaR")
library("caret")

model.nb.cv <- train(Train_mod[c(2:12,15:17)],Train_mod$Loan_Status, trControl = trainControl(method="cv",number=10))

predict(model.nb.cv$finalModel,Train_mod[c(2:12,15:17)],type="prob")


















