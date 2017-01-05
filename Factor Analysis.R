us.pollution <- read.csv("C:/Users/Sandhya/OneDrive/Data Science Practical Learning/Data/us pollution.csv")

Num_Vars <- us.pollution[c(6:9,11:14,16:18,21:23)]
Num_Vars <- scale(Num_Vars)
# Determine Number of Factors to Extract
library(nFactors)
ev <- eigen(cor(Num_Vars)) # get eigenvalues
ap <- parallel(subject=nrow(Num_Vars),var=ncol(Num_Vars),
               rep=100,cent=.05)
nS <- nScree(x=ev$values, aparallel=ap$eigen$qevpea)
plotnScree(nS)




















