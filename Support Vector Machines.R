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

library("e1071", lib.loc="~/R/win-library/3.3")
model.svm <- svm(Loan_Status ~ Gender+Married+Dependents+Education+Self_Employed+
                   LoanAmount+Loan_Amount_Term+
                   Credit_History+Property_Area+Income+share+Co_App, data = Train_mod, cost = 100, gamma = 1)


Train_mod$svm.pred <- predict(model.svm, Train_mod)

Train_mod$svm.pred  <- ifelse(Train_mod$svm.pred =="Y",1,0)

conf_Mat <- table(Train_mod$Loan_Status , Train_mod$svm.pred)


# Tuning svm 
# perform a grid search
tuneResult <- tune(svm, Loan_Status ~ Gender+Married+Dependents+Education+Self_Employed+
                     LoanAmount+Loan_Amount_Term+
                     Credit_History+Property_Area+Income+share+Co_App,  data = Train_mod,
                   ranges = list(epsilon = seq(0,1,0.1), cost = 2^(2:9))
)
print(tuneResult)
# best performance: MSE = 8.371412, RMSE = 2.89 epsilon 1e-04 cost 4
# Draw the tuning graph
plot(tuneResult)
tunedModel <- tuneResult$best.model
tunedModelY <- predict(tunedModel, Train_mod)
tunedModelY  <- ifelse(tunedModelY =="Y",1,0)

conf_Mat <- table(Train_mod$Loan_Status , tunedModelY)
# this model doesnt peform better than the above model

# diff parameters to tune
tuneResult <- tune(svm, Loan_Status ~ Gender+Married+Dependents+Education+Self_Employed+
                     LoanAmount+Loan_Amount_Term+
                     Credit_History+Property_Area+Income+share+Co_App,  data = Train_mod,
                   kernel="radial", ranges=list(cost=10^(-1:2), gamma=c(.5,1,2)))
print(tuneResult)
# best performance: MSE = 8.371412, RMSE = 2.89 epsilon 1e-04 cost 4
# Draw the tuning graph
plot(tuneResult)
tunedModel <- tuneResult$best.model
tunedModelY <- predict(tunedModel, Train_mod)
tunedModelY  <- ifelse(tunedModelY =="Y",1,0)

conf_Mat <- table(Train_mod$Loan_Status , tunedModelY)




# For probability outputs - https://cran.r-project.org/web/packages/e1071/e1071.pdf
x <- Train_mod[c(2:13,15:17)]
x <- subset(x,select = -Loan_Status)
attach(x)
model.svm1 <- svm(x,Loan_Status, probability = TRUE)
Train_mod$svm.pred1 <- predict(model.svm1, Train_mod)
