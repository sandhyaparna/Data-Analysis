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


# Only Numerical data - encoding is done on categorical data

fitControl <- trainControl(method = "repeatedcv", number = 4, repeats = 4)
#Train_mod$outcome1 <- ifelse(trainData$Disbursed == 1, "Yes","No")
set.seed(33)
gbmFit1 <- train(as,factor(Loan_Status) ~ 
                   LoanAmount+Loan_Amount_Term+
                   Property_Area+Income+share, data = Train_mod[c(2:12,15:17)], method = "gbm", trControl = fitControl,verbose = FALSE)




