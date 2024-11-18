library(data.table)
library(caret)

# load the file you need
x <- fread("classification_train-L2-7B-cap-verified-v2.csv")
x <- fread("classification_test-L2-7B-cap-verified-v2.csv")
x <- fread("classification_test-L2-7B-cap-verified-last-and-final.csv")
x <- fread("classification_train-L2-7B-cap-verified-last-and-final.csv")

round(prop.table(table(x$isCorrect))*100,1)
dim(x)

tab <- table(x$predicted, x$trueCat)
cats <- as.character(sort(unique(x$trueCat)))
ncats <- length(cats)
mat <- matrix(0, ncats, ncats)
colnames(mat) <- rownames(mat) <- cats
for(i in 1:ncats){
    for(j in 1:ncats){
        try(mat[cats[i],cats[j]] <- tab[cats[i],cats[j]], TRUE)
    }
}

cmMat <- confusionMatrix(mat , mode="everything" )
cmMat
str(cmMat)
st <- round(t(cmMat$byClass[,c("Sensitivity", "Specificity", "F1", "Balanced Accuracy")])*100,1)
st
round(apply(st, 1, mean,na.rm=TRUE),1)
