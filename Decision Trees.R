# http://www.listendata.com/2014/11/random-forest-with-r.html
# https://www.r-bloggers.com/a-brief-tour-of-the-trees-and-forests/
# https://www.analyticsvidhya.com/blog/2016/04/complete-tutorial-tree-based-modeling-scratch-in-python/
# http://machinelearningmastery.com/non-linear-classification-in-r-with-decision-trees/  
library("rpart", lib.loc="C:/Program Files/R/R-3.3.1/library")

train_LP <- read.csv("C:/Users/Sandhya/OneDrive/Data Science Practical Learning/Projects/Loan Prediction/train_LP.csv")
train_LP$Data <- "Train"
test_LP <- read.csv("C:/Users/Sandhya/OneDrive/Data Science Practical Learning/Projects/Loan Prediction/test_LP.csv")
test_LP$Loan_Status <- NA
test_LP$Data <- "Test"

All <- rbind(train_LP,test_LP)

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
All$Co_App <- factor(ifelse(All$CoapplicantIncome==0,"No","Yes"))

Train_mod <- dplyr::filter(All,Data=="Train")
Test_mod <- dplyr::filter(All,Data=="Test")

# Regression Tree will have "anova" option in method
# control=rpart.control(minsplit=30, cp=Model.CART$cptable[which.min(Model.CART$cptable[,"xerror"]),"CP"])
# control option is used to control tree growth
Model.CART <- rpart(Loan_Status~Gender+Married+Dependents+Education+Self_Employed+
                      LoanAmount+Loan_Amount_Term+
                      Credit_History+Property_Area+Income+share+Co_App,
                    data=Train_mod, method="class" )

# 	display cp table
printcp(Model.CART)

# plot cross-validation results
plotcp(Model.CART)

# plot tree 
plot(Model.CART, uniform=TRUE,main="Classification Tree")
text(Model.CART, use.n=TRUE, all=TRUE, cex=.8)

# create attractive postscript plot of tree 
post(Model.CART, file = "C:/Users/Sandhya/OneDrive/Data Science Practical Learning/tree.ps", 
     title = "Classification Tree")

# Print results
print(Model.CART)

# prune the tree - Is based on cross validation
pfit<- prune(Model.CART, cp=   Model.CART$cptable[which.min(Model.CART$cptable[,"xerror"]),"CP"])

# plot the pruned tree 
plot(pfit, uniform=TRUE, 
     main="Pruned Classification Tree")
text(pfit, use.n=TRUE, all=TRUE, cex=.8)

a <- as.data.frame(predict(Model.CART))
predict(pfit)

Train_mod$pred <- as.data.frame(predict(Model.CART))$Y
Train_mod$pred1 <- predict(Model.CART, type = "class")
Train_mod$pred <- ifelse(Train_mod$pred>0.5,1,0)
conf_Mat <- table(Train_mod$Loan_Status , Train_mod$pred)

#ROC Curve
library(ROCR)
ROCRpred <- prediction(Train_mod$pred, Train_mod$Loan_Status)
ROCRperf <- performance(ROCRpred, 'tpr','fpr')
plot(ROCRperf, colorize = TRUE, text.adj = c(-0.2,1.7))

#AUC - Area  under the curve
library(pROC)
auc(Train_mod$Loan_Status, Train_mod$pred)

# TREE package
library(tree)
Model.tr = tree::tree(Loan_Status~Gender+Married+Dependents+Education+Self_Employed+
            LoanAmount+Loan_Amount_Term+
            Credit_History+Property_Area+Income+share+Co_App,
          data=Train_mod)
summary(Model.tr)
plot(Model.tr)
text(Model.tr)

url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data"
crx <- read.table( file=url, header=FALSE, sep="," )

# C5.0 algorithm for Decision Trees
#http://www.patricklamle.com/Tutorials/Decision%20tree%20R/Decision%20trees%20in%20R%20using%20C50.html

library("C50", lib.loc="~/R/win-library/3.3")
#model <- C50::C5.0( trainX, trainy )
model.C5 <- C50::C5.0(Train_mod[c(2:12,15:17)], Train_mod$Loan_Status)
summary(model.C5)
model.C5
Train_mod$pred <- predict(model.C5,Train_mod,type="prob")

# Boosting the accuracy of decision trees - Adaboost, add trails option
credit_boost10 <- C5.0(credit_train[-17], credit_train$default,
                       trials = 10)
credit_boost_pred10 <- predict(credit_boost10, credit_test)


# Making some mistakes more costly than others

error_cost <- matrix(c(0, 1, 4, 0), nrow = 2) # create a cost matrix

credit_cost <- C5.0(credit_train[-17], credit_train$default,
                    costs = error_cost) # Apply the cost matrix to the tree

credit_cost_pred <- predict(credit_cost, credit_test)


#C5.0 Cross validation
# http://www.euclidean.com/machine-learning-in-practice/2015/6/12/r-caret-and-parameter-tuning-c50


#http://machinelearningmastery.com/non-linear-classification-in-r-with-decision-trees/

#C4.5 algorithm RWeka::J48
library("RWeka", lib.loc="~/R/win-library/3.3")
library("party", lib.loc="~/R/win-library/3.3")
Model.C4.5 <- J48(Loan_Status~Gender+Married+Dependents+Education+Self_Employed+
                           LoanAmount+Loan_Amount_Term+
                           Credit_History+Property_Area+Income+share+Co_App,
                         data=Train_mod)
summary(Model.C4.5)
Model.C4.5


# Random Forest 
Model.RandomForest <- randomForest::randomForest(Loan_Status~Gender+Married+Dependents+Education+Self_Employed+
                                                   LoanAmount+Loan_Amount_Term+
                                                   Credit_History+Property_Area+Income+share+Co_App,
                  data=Train_mod)
summary(Model.RandomForest)
Model.RandomForest
Train_mod$pred <- predict(Model.RandomForest,type="prob")

# change values of ntree
Model.RandomForest <- randomForest::randomForest(Loan_Status~Gender+Married+Dependents+Education+Self_Employed+
                                                   LoanAmount+Loan_Amount_Term+
                                                   Credit_History+Property_Area+Income+share+Co_App,
                                                 data=Train_mod, ntree=200)
summary(Model.RandomForest)
Model.RandomForest

# change values of ntree
Model.RandomForest <- randomForest::randomForest(Loan_Status~Gender+Married+Dependents+Education+Self_Employed+
                                                   LoanAmount+Loan_Amount_Term+
                                                   Credit_History+Property_Area+Income+share+Co_App,
                                                 data=Train_mod, ntree=100)
summary(Model.RandomForest)
Model.RandomForest

