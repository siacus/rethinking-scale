# This scripts generates the stats about the correct classification
# for each fine-tuned or plain model
# (S.M.Iacus 2024)

library(data.table)
library(jsonlite)
library(knitr)
library(kableExtra)

# pick the file you need for your analysis

# the fine-tuned versions
#x <- fread("classification_results-llama-2-7b-dv.csv")
#x <- fread("classification_results-llama-2-7b-small-dv.csv")

# The plan versions
#x <- fread("classification_results-7B-NO-FT.csv")
#x <- fread("classification_results-13B-NO-FT.csv")
#x <- fread("classification_results-70B-NO-FT.csv")

prop.table(table(x$isCorrect))

# we transform empty classification in NA
x$predicted[x$predicted %in% c("[]", '"[]"', '[', "['[']")] <- "['N/A']" 

# this function extract a str vector from a python list
f <- function(tmp){
    tmp <- gsub("'", '"', tmp)
    tmp <- fromJSON(tmp)
    tmp
}


nTrue <- sapply(x$trueSubject, function(u) length(f(u)), USE.NAMES = FALSE)
x$nTrue <- nTrue

# the list of true categories. Includes N/A and Other
trueCat <- sort(unique(unlist(sapply(x$trueSubject, f))))
trueCat

# we transform wrong classification
tt <- sapply(x$predicted, function(u) sum(unlist(sapply(trueCat, function(h) grep(h,u)))))
table(tt)
idx <- which(tt == 0)
#tt[idx]
x$predicted[idx] <- "['N/A']"
tt <- sapply(x$predicted, function(u) sum(unlist(sapply(trueCat, function(h) grep(h,u)))))
table(tt)

# at this point at least one subject category is present in the prediction or N/A
# this means that there might be made of categories along with true categories

# now we look into wrong classifications
idx <- which(!x$isCorrect)
length(idx)

# some are wrong because we extraced them wrongly, so I need to fix it
# > x$predicted[x$id=='doi:10.7910/DVN/DULFFJ']
# [1] "['Arts and Humanities, Social Sciences']"

# we cleanup further errors 
g <- function(tmp){
    tmp <- names(unlist(sapply(trueCat, function(u) grep(u,tmp,fixed=TRUE))))   
    tmp <- paste0("[", paste(shQuote(tmp, type = "cmd"), collapse = ", "), "]")
}

# this is the cleaned version
x$predictedClean <- sapply(x$predicted, g, USE.NAMES = FALSE)

nPred <- sapply(x$predictedClean, function(u) length(f(u)), USE.NAMES = FALSE)
x$nPred <- nPred
table(x$nTrue, x$nPred)

# test
# x[x$id=='doi:10.7910/DVN/DULFFJ',]

# now we fix isCorrect
tt <- sapply(1:nrow(x), function(i) length(intersect(f(x$trueSubject[i]),f(x$predictedClean[i]))))

1-length(which(tt==0))/sum(table(tt))
round(prop.table(table(x$isCorrect))*100,1)

kable(round(prop.table(table(x$nTrue, x$nPred))*100,1), format = "latex", booktabs = FALSE)
