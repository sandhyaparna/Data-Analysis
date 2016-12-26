train_LP <- read.csv("C:/Users/Sandhya/OneDrive/Data Science Practical Learning/Projects/Loan Prediction/train_LP.csv")
train_LP$Data <- "Train"
test_LP <- read.csv("C:/Users/Sandhya/OneDrive/Data Science Practical Learning/Projects/Loan Prediction/test_LP.csv")
test_LP$Loan_Status <- NA
test_LP$Data <- "Test"
summary(train_LP)
summary(test_LP)

All <- rbind(train_LP,test_LP)
# we can see a lot of missing values 
credit_Miss <- dplyr::filter(train_LP,is.na(Credit_History))
credit_Miss_Freq <- as.data.frame(table(credit_Miss$Loan_Status))
# We can see out of 50 missing Y and N are split in ratio of 2:1 - We can say that the mssiing values are random
#Perform 2 freq table of Credit_history and Loan_status 
# From the freq table we can see that a perform whose loan status  
credit <- xtabs(~ Credit_History  + Loan_Status, train_LP)






# If credit_history is missing - Create a new Category for it
All$Credit_History <- factor(ifelse(is.na(All$Credit_History),"2",as.character(All$Credit_History)))
All$Gender <- factor(ifelse(All$Gender=="","Blank",as.character(All$Gender)))
All$Married <- factor(ifelse(All$Married=="","Blank",as.character(All$Married)))
All$Dependents <- factor(ifelse(All$Dependents=="","Blank",as.character(All$Dependents)))
All$Self_Employed <- factor(ifelse(All$Self_Employed=="","Blank",as.character(All$Self_Employed)))
LoanAmt <- aggregate(LoanAmount ~ Loan_Status , All, mean)
# Average Loan amounts for Y and N is almost the same - so impute Missing Loan amounts with Overall Mean
All$LoanAmount <- ifelse(is.na(All$LoanAmount),142.5,All$LoanAmount)
LoanAmtTerm <- aggregate(Loan_Amount_Term ~ Loan_Status , All, mean)
# Average Loan_Amount_Term for Y and N is almost the same - so impute Missing Loan_Amount_Term with Overall Mean
All$Loan_Amount_Term <- ifelse(is.na(All$Loan_Amount_Term),342.2,All$Loan_Amount_Term)

All$Income <- All$ApplicantIncome+All$CoapplicantIncome

All$share <- All$LoanAmount/All$Income
All$Co_App <- ifelse(All$CoapplicantIncome==0,"No","Yes")

Train_mod <- dplyr::filter(All,Data=="Train")
Test_mod <- dplyr::filter(All,Data=="Test")

summary(All)
model <- glm(formula = Loan_Status ~ Gender+Married+Dependents+Education+Self_Employed+
               LoanAmount+Loan_Amount_Term+
               Credit_History+Property_Area+Income+share+Co_App, family = binomial(link = "logit"), 
    data = Train_mod)
summary(model)


model1 <- glm(formula = Loan_Status ~ Dependents+Married+Education+Credit_History+Property_Area+Income+share+Co_App, family = binomial(link = "logit"), 
             data = Train_mod)
summary(model1)

model2 <- glm(formula = Loan_Status ~ Credit_History+Property_Area+Income+share, family = binomial(link = "logit"), 
              data = Train_mod)
summary(model2)

# Variable Importance
caret::varImp(model)
Train_mod$pred <- predict(model2)
Train_mod$pred <- predict(model2,type = "response")
Train_mod$pred <- ifelse(Train_mod$pred>0.57,1,0)
conf_Mat <- table(Train_mod$Loan_Status , Train_mod$pred1)

#ROC Curve
library(ROCR)
ROCRpred <- prediction(Train_mod$pred, Train_mod$Loan_Status)
ROCRperf <- performance(ROCRpred, 'tpr','fpr')
plot(ROCRperf, colorize = TRUE, text.adj = c(-0.2,1.7))

#AUC - Area  under the curve
library(pROC)
auc(Train_mod$Loan_Status, Train_mod$pred)

Test_mod$pred2 <- predict(model2, newdata = Test_mod, type = "response")


# Cross validation
train_control <- trainControl(method="cv", number=10)
model.cv <- train(Loan_Status ~ Credit_History+Property_Area+Income+share
                  , data=Train_mod, trControl=train_control)
summary(model.cv)
Train_mod$pred.cv <- predict(model.cv)

# For a different threshold
Train_mod$pred.cv_thres <- predict(model.cv,type = "prob")

# Model evaluation - Cross validation
#Train_mod$pred.cv <- ifelse(Train_mod$pred3<0.5,1,0)
conf_Mat <- table(Train_mod$Loan_Status , Train_mod$pred.cv)
Train_mod$pred.cv <- ifelse(Train_mod$pred.cv=="Y",1,0)

#ROC Curve
library(ROCR)
ROCRpred <- prediction(Train_mod$pred.cv, Train_mod$Loan_Status)
ROCRperf <- performance(ROCRpred, 'tpr','fpr')
plot(ROCRperf, colorize = TRUE, text.adj = c(-0.2,1.7))

#AUC - Area  under the curve
library(pROC)
auc(Train_mod$Loan_Status, Train_mod$pred.cv)

# Predict on Test data
Test_mod$pred.cv <- predict(model.cv, newdata = Test_mod)


# Wald Test?



# To determine optimal polynomial degree 
glm.fit = glm(Loan_Status ~ Credit_History, family = binomial(link = "logit"), data=Train_mod)
degree=1:5
cv.error5=rep(0,5)
for(d in degree){
  glm.fit = glm(Loan_Status ~ Credit_History,family = binomial(link = "logit"),  data=Train_mod)
  cv.error5[d] = cv.glm(Train_mod,glm.fit,K=5)$delta[1]
}
plot(degree,cv.error5)

+Property_Area+Income+share



