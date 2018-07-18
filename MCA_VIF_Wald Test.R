require(FactoMineR)
require(ggplot2)

# load data tea
data(tea)

# select these columns
newtea = tea[, c("Tea", "How", "how", "sugar", "where", "always")]

# take a peek
head(newtea)

# number of categories per variable
cats = apply(newtea, 2, function(x) nlevels(as.factor(x)))

cats

# apply MCA
mca1 = MCA(newtea, graph = FALSE)

# list of results
mca1

# table of eigenvalues
mca1$eig

mca1$var

factors <- names(Sepsis_Jan17_Mar18_AllVitals_Gre2days[c(16:71,212:267)])
as.formula(paste("Sepsis~", paste(factors, collapse="+")))


# Efficient way to calculating VIF - run vif , var with highest 
#vif value is removed and then re-run vif  in this process till all vars have vif < 10
All[is.na(All)] <- 0
fit <- lm(share ~ ApplicantIncome+CoapplicantIncome+Loan_Amount_Term, data=All)
vif(fit)

#
VIF {fmsb}
res <- lm(share ~ ApplicantIncome+CoapplicantIncome+Loan_Amount_Term, data=All)
VIF(lm(ApplicantIncome~CoapplicantIncome+Loan_Amount_Term, data=All))
VIF(lm(CoapplicantIncome ~ ApplicantIncome+Loan_Amount_Term, data=All))
VIF(lm(Loan_Amount_Term ~ ApplicantIncome+CoapplicantIncome, data=All))

library(aod)
library(ggplot2)
library(Rcpp)
model1 <- glm(formula = Loan_Status ~ Dependents+Married+Education+Credit_History+Property_Area+Income+share+Co_App, family = binomial(link = "logit"), 
              data = Train_mod)
summary(model1)

# Wald Test - to identify insignificant variables in logistic regression
# 2:4 vars are output vars from logistic regression - belongs to Dependent variable above
# Wald test has to be done for each variable
#http://www.ats.ucla.edu/stat/r/dae/logit.htm
wald.test(b = coef(model1), Sigma = vcov(model1), Terms = 2:4)
