# Performs the cleaning and analysis on the classifications
# (S.M.Iacus 2024)

rm(list=ls())

categories <- tolower(c(
"Happiness", 
"Resilience", 
"Self-esteem", 
"Life satisfaction", 
"Fear of future", 
"Vitality", 
"Having energy", 
"Positive functioning", 
"Expressing job satisfaction", 
"Expressing optimism", 
"Peace with thoughts and feelings",
"Purpose in life", 
"Depression", 
"Anxiety", 
"Suffering", 
"Feeling pain",
"Expressing altruism", 
"Loneliness", 
"Quality of relationships", 
"Belonging to society", 
"Expressing gratitude", 
"Expressing trust", 
"Feeling trusted", 
"Balance in the various aspects of own life", 
"Mastery (ability, capability)", 
"Perceiving discrimination", 
"Feeling loved by God", 
"Belief in God", 
"Religious criticism", 
"Spiritual punishment", 
"Feeling religious comfort",
"Financial/material worry", 
"Life after death belief", 
"Volunteering", 
"Charitable giving/helping", 
"Seeking for forgiveness", 
"Feeling having a political voice", 
"Expressing government approval", 
"Having hope", 
"Promoting good", 
"Expressing delayed gratification",
"PTSD (Post-traumatic stress disorder)", 
"Describing smoking related health issues", 
"Describing drinking related health issues", 
"Describing health limitations", 
"Expressing empathy"))

files_to_delete <- list.files(pattern = "^summ8-.*")
file.remove(files_to_delete)

removeLow <- FALSE
FNW <- 0.1

library(data.table)
true <- fread("all-llms-codings.csv", colClasses = 'character'  )

true$accepted[true$accepted == "Y"] <- "1"  # just for symmetry with the other script
true$accepted[true$accepted == "N"] <- "0"  # just for symmetry with the other script

true <- data.frame(true)
true[which(true$id=="1632974707741048832" & true$accepted=="1"),]

badIds <- c("1630780767327469570",
"1629215331444215808",
"1625975958372057113",
"1625826146670747649",
"1609351093795917832",
"1629906682087108609",
"1628439619669839872",
"1628867674833932293",
"1630925537592893440",
"1631200067183935491",
"1627051273949413377",
"1628770482429239297"
)

ncat <- length(categories)

idx <- which((true$id %in% badIds) & true$coder == "10")
if(length(idx)>0){
    true <- true[-idx,]
}
true[which(true$id=="1632974707741048832" & true$accepted=="1"),]

true <- true[true$accepted == "1",]
dups <- which(duplicated(true[,c("id","dimension","value","accepted")]))
ndups <- length(dups)
ndups
if(ndups>0){
    true <- true[-dups,]
}

# now we check for duplicated dimensions. In this case we use
# the criteria 
# if(low & medium) we choose low
# otherwise if(low & high or medium & high) we choose high 
dups <- which(duplicated(true[,c("id","dimension","accepted")]))
ndups <- length(dups)
ndups
if(ndups>0){
    for(i in 1:ndups){
        id <- true$id[dups[i]]
        dim <- true$dim[dups[i]]
        # if there is a "high" we transform everything to "high"
        if("high" %in% true$value[true$id == id & true$dim == dim]){
            true$value[true$id == id & true$dim == dim] <- "high"
        } 
        # if there wasn't a "high", we check for "medium"
        if("medium" %in% true$value[true$id == id & true$dim == dim]){
            true$value[true$id == id & true$dim == dim] <- "medium"
        } # if there wass a "medium", all should be "medium" now
        # the next case should never happen
        if("low" %in% true$value[true$id == id & true$dim == dim]){
            true$value[true$id == id & true$dim == dim] <- "low"
        }
    }
} # at the end we have created duplicated, so we need to remove them again

dups <- which(duplicated(true[,c("id","dimension","value","accepted")]))
ndups <- length(dups)
ndups
if(ndups>0){
    true <- true[-dups,]
}

dups <- which(duplicated(true[,c("id","dimension","accepted")]))
ndups <- length(dups)
ndups # should be 0

true[which(true$id=="1632974707741048832" & true$accepted=="1"),]

if(removeLow){
    lows <- which(true$value == 'low')
    if(length(lows)>0){
        true <- true[-lows,]
    }
}



models <- fread("models.csv", sep=',')
nmod <- nrow(models)

for(modNum in 1:nmod){
    FT <- models$FT[modNum]
    MODEL <-  models$MODEL[modNum]
    BPAR <- models$BPAR[modNum]
    fname <- models$fname[modNum]
    print(models[modNum,1:3])
    classified <- fread(sprintf("%s.csv",fname), colClasses = 'character'  )
    source("clean-classified.R")
    if(removeLow){
        clows <- which(classified$value == 'low')
        if(length(clows)>0){
            classified <- classified[-clows,]
        }
    }
    dt <- true
    dt <- as.data.frame(dt)
    if(removeLow){
        dtlows <- which(dt$value == 'low')
        if(length(dtlows)>0){
            dt <- dt[-dtlows,]
        }
    }

    common_ids <- intersect(true$id[true$accepted=="1"],classified$id)
    if(length(common_ids)>0){

    ids = unique(common_ids)
    nids = length(ids)

    tmp <- NULL
    tmpD <- NULL
    jac <- NULL
    jac2 <- NULL
    ham <- NULL

    for(i in 1:nids){
        tmpD2 <- NULL
        # next 3 lines only to calculate the Jaccard index
        true_dim <- true$dimension[true$id == ids[i] & true$accepted=="1"]
        true_val <- true$value[true$id == ids[i] & true$accepted=="1"]
        
        llm_dim <- classified$dimension[classified$id == ids[i]]
        llm_val <- classified$value[classified$id == ids[i]]
        
        true_positives <- intersect(true_dim,llm_dim)
        jac <- c(jac, length(true_positives) / length(union(true_dim,llm_dim)))

        true_negatives <- union(setdiff(categories, llm_dim), setdiff(llm_dim, categories))
        true_negatives <- true_negatives[!(true_negatives %in% true_dim)]
        
        nTN <- length(true_negatives)
        if(nTN>0){
            tmp <- rbind(tmp, cbind(rep("not present", nTN), rep("not present", nTN), rep(ids[i], nTN)))
            labNeg <- sprintf("not:%s", true_negatives)
            tmpD <- rbind(tmpD, cbind(labNeg, labNeg, rep(ids[i], nTN)))
            tmpD2 <- rbind(tmpD2, cbind(labNeg, labNeg, rep(ids[i], nTN)))
        }
        
        idxTP <- which(true_dim %in% llm_dim) # LLM get positives right
        nTP <- length(idxTP)
        if(nTP>0){
            tmp <- rbind(tmp, cbind(true_val[idxTP], llm_val[match(true_dim[idxTP], llm_dim)], rep(ids[i], nTP)))
            tmpD <- rbind(tmpD, cbind(true_dim[idxTP], llm_dim[match(true_dim[idxTP], llm_dim)], rep(ids[i], nTP)))
            tmpD2 <- rbind(tmpD2, cbind(true_dim[idxTP], llm_dim[match(true_dim[idxTP], llm_dim)], rep(ids[i], nTP)))
        }
        idxFN <- which(!(true_dim %in% llm_dim)) # LLM misses true positives
        nFN <- length(idxFN)
        if(nFN>0){
            tmp <- rbind(tmp, cbind(true_val[idxFN], rep("not present", nFN), rep(ids[i], nFN)))
            tmpD <- rbind(tmpD, cbind(true_dim[idxFN], rep("not present", nFN), rep(ids[i], nFN)))
            tmpD2 <- rbind(tmpD2, cbind(true_dim[idxFN], rep("not present", nFN), rep(ids[i], nFN)))
        }
        idxFP <- which(!(llm_dim %in% true_dim)) # LLM misses true positives
        nFP <- length(idxFP)
        if(nFP>0){
            dups <- which(duplicated(llm_dim[idxFP]))
            if(length(dups)>0){
                idxFP <- idxFP[-dups]
            }
            nFP <- length(idxFP)
            tmp <- rbind(tmp, cbind(rep("not present", nFP), llm_val[idxFP], rep(ids[i], nFP)))
            tmpD <- rbind(tmpD, cbind(rep("not present", nFP), llm_dim[idxFP], rep(ids[i], nFP)))
            tmpD2 <- rbind(tmpD2, cbind(rep("not present", nFP), llm_dim[idxFP], rep(ids[i], nFP)))
        }
        ham <- c(ham, length(which(tmpD2[,1] != tmpD2[,2]))/ncat)
        jac2 <- c(jac2, length(which(tmpD2[,1] == tmpD2[,2]))/length(union(tmpD2[,1],tmpD2[,2])) )
        if(nTN+nTP+nFN+nFP != 46) stop() # something wrong
    }


    colnames(tmp) <- c("true", "llm", "id")
    tmp <- data.frame(tmp, stringsAsFactors = FALSE)

    colnames(tmpD) <- c("true", "llm", "id")
    tmpD <- data.frame(tmpD, stringsAsFactors = FALSE)


    valuesc <- c("low", "medium", "high", "not present")
    valuesr <- c( "low", "medium", "high",  "not present")             

    tab <- table(tmp[,1:2])
    sum(tab)

    tabD <- table(tmpD[,1:2])
    sum(tabD)
    
    tabb <- matrix(0, 4, 4)
    colnames(tabb) <-  valuesc
    rownames(tabb) <-  valuesr

    for(i in valuesr){
        for(j in valuesc){
            t <- try(tab[i, j], TRUE)
            if(class(t)!='try-error'){
                tabb[i,j] <- t
            }
        }
    }
    tab <- as.table(tabb)
    tab

    labNeg <- sprintf("not:%s", categories)
    dimc <- c(categories, labNeg, "not present")
    dimr <- c(categories, labNeg, "not present")             
    tabbD <- matrix(0, length(dimr), length(dimc))

    colnames(tabbD) <-  dimc
    rownames(tabbD) <-  dimr

    for(i in dimr){
        for(j in dimc){
            t <- try(tabD[i, j], TRUE)
            if(class(t)!='try-error'){
                tabbD[i,j] <- t
            }
        }
    }
    tabD <- as.table(tabbD)


    acc <- NULL
    f1 <- NULL
    prec <- NULL
    recall <- NULL
    spec <- NULL
    sens <- NULL
    balAcc <- NULL
    for(i in 1:ncat){
        mycat <- categories[i]
        mycatNot <- sprintf("not:%s",mycat)
        allcats <- c(mycat, mycatNot, "not present")
        tmpTab <- tabD[allcats,allcats]
        tmpTab[mycatNot, mycat] <-  tmpTab["not present",mycat]
        tmpTab[mycat, mycatNot] <-  tmpTab[mycat, "not present"]
        tmpTab <- tmpTab[c(mycat,mycatNot),c(mycat,mycatNot)]
        cm <- caret::confusionMatrix(tmpTab,mode="everything", positive = mycat)
        acc <- c(acc, cm$overall["Accuracy"])
        f1 <- c(f1, cm$byClass["F1"])
        prec <- c(prec, cm$byClass["Precision"])
        recall <- c(recall, cm$byClass["Recall"])
        spec <- c(spec, cm$byClass["Specificity"])
        sens <- c(sens, cm$byClass["Sensitivity"])
        balAcc <- c(balAcc, cm$byClass["Balanced Accuracy"])
    }

    names(acc) <- categories
    names(f1) <- categories
    names(prec) <- categories
    names(recall) <- categories
    names(spec) <- categories
    names(sens) <- categories
    names(balAcc) <- categories
    
    stats <- data.frame(dimension=categories, acc = acc, f1 = f1, prec = prec,
                recall = recall, spec = spec, sens = sens, balAcc = balAcc, stringsAsFactors = FALSE)



    ntext = nids
  
    n = sum(tabD)
    ptab <- round(tab/n,2)
    ptab
    labs <- c('low','medium','high')

    accD <- sum(diag(tabD))/sum(tabD) 
    accI <- sum(diag(tab))/sum(tab)

    summ <- NULL
    summ <- data.frame(model=MODEL,  bpar = BPAR, ft=FT, type="FULL", 
    n=n, 
    ntext = ntext,
    accD = accD,
    accI = accI, 
    FN = sum(tabD[categories,"not present"])/n, 
    FP = sum(tabD["not present",categories])/n, 
    jac = mean(jac),
    jac2 = mean(jac2),
    ham = mean(ham),
    fname=fname, stringsAsFactors = FALSE)


    summ2 <- NULL
    summ2 <- data.frame(stats, model=MODEL,  bpar = BPAR, ft=FT, type="FULL", 
    n=n, 
    ntext = ntext,
    accD = accD,
    accI = accI, 
    FN = sum(tabD[categories,"not present"])/n, 
    FP = sum(tabD["not present",categories])/n, 
    jac = mean(jac),
    jac2 = mean(jac2),
    ham = mean(ham),
    fname=fname, 
    stringsAsFactors = FALSE)
    rownames(summ2) <- NULL

    cat(sprintf("\nFile = %s\nN=%d\n",fname, sum(tab)))
    print(tab)
    print(ptab)
    cat(sprintf("\nAccuracy intensity: %.1f%%\n", 100*accI))
    cat(sprintf("\nAccuracy dimensions: %.1f%%\n", 100*accD))
    cat(sprintf("\nJaccard index: %.1f\n", mean(jac)))
    cat(sprintf("\nJaccard index 2: %.1f\n", mean(jac2)))
    cat(sprintf("\nHamming loss: %.1f\n======================================\n\n", mean(ham)))

    write.csv(summ, file=sprintf("summ8-%s.csv",fname), row.names=FALSE)
    write.csv(summ2, file=sprintf("stats8-%s.csv",fname), row.names=FALSE)
  
    print(summ)
     } else {
          cat(sprintf("\t model %s skipped\n", MODEL))
    }
}



files <- list.files(path = ".",pattern = "summ8-")

l <- lapply(files, fread, sep=",")
dt <- rbindlist( l )

write.csv(dt, file="summaryOld.csv", row.names= FALSE)
files_to_delete <- list.files(pattern = "^summ8-.*")
file.remove(files_to_delete)


files2 <- list.files(path = ".",pattern = "stats8-")

l2 <- lapply(files2, fread, sep=",")
dt2 <- rbindlist( l2 )

write.csv(dt2, file="summaryStatsOld.csv", row.names= FALSE)
files_to_delete <- list.files(pattern = "^stats8-.*")
file.remove(files_to_delete)

