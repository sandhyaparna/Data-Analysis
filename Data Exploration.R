offers <- read.csv("E:/R Learning/offers.csv")

titanic3 <- read.csv("C:/Users/Sandhya/Downloads/titanic3.csv")

M <- titanic3[c(5,9)]
M <- na.omit(M)
corrplot::corrplot(M, method="number")

fl <- fl2000[c(4:17)]

library(corrplot)
M <- cor(fl)
corrplot(M, method="circle")

library(Hmisc)
Hmisc::rcorr(fl, type="pearson") 
cor(fl, use="complete.obs", method="kendall") 
cov(fl, use="complete.obs")