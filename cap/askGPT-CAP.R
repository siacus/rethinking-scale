# Calling OpenAI APIs to do the preliminary classification
# (S.M.Iacus 2024)

rm(list=ls())
library(jsonlite)
library(httr)
library(stringr)
library(tidyverse)
library(writexl)

# use your own OpenAI API token
openai_api_key <- Sys.getenv("OPENAI_API_KEY", "sk-YOUR_OPENAI_TOKEN") 


askGPT <- function(what, text, hint = "Sentiment", v4=FALSE, verbose=TRUE, temperature=0.0){
  prompt <- sprintf("%s\nText:\"%s\"%s", what, text, hint)
  if(verbose)
    cat(prompt)
  response <- POST(
    url = "https://api.openai.com/v1/chat/completions", 
    add_headers(Authorization = paste("Bearer", openai_api_key)),
    content_type_json(),
    encode = "json",
    body = list(
      model = ifelse(v4,"gpt-4", "gpt-3.5-turbo-0125"), # fix this to the current version of ChatGPT
      messages = list(list(role = "user", content = prompt))
    ) 
  )
  answer <- content(response)
  if(verbose)
    cat(sprintf("\n%s\n", answer$choices[[1]]$message$content))
  return(invisible(answer))
}

dat <- xlsx::read.xlsx("cap_training_set.xlsx", sheetIndex = 1)  
codebook <- xlsx::read.xlsx("cap_codebook.xlsx", sheetIndex = 1)

dat$topic_major <- as.character(NA)

idx.minor <- match(dat$minor.topic, codebook$subtopic) 
dat$topic_minor <- as.character(NA)

# we fix for a few NA's
if(length(idx.minor)>0){
  head(which(is.na(idx.minor)))
  dat$topic_minor <-  codebook$label.1[idx.minor]
  dim(dat)
  dat <- dat[-which(is.na(dat$topic_minor)), ]
} 
  

major_topics <- unique(codebook$major_topic_label)
minor_topics <- unique(codebook$subtopic_label)
minor_topics_short <- unique(codebook$subtopic_label_short)

major_topics_num <- codebook$major[match(major_topics,codebook$major_topic_label)]
minor_topics_num <- codebook$subtopic[match(minor_topics,codebook$subtopic_label)]
minor_topics_short_num <- codebook$subtopic[match(minor_topics_short,codebook$subtopic_label_short)]


Nt <- length(major_topics)
nt <- length(minor_topics)
nlt <- length(minor_topics_short)

N <- 1001

what <- 'The next text is a question of a member of the European parliament. Classify the question according to only one of the policy categories below. Use only numbers to answer'



minor_mat <- matrix(0,nrow = N, ncol=Nt)
colnames(minor_mat) <- major_topics

major_mat <- matrix(0,nrow = N, ncol=Nt)
colnames(major_mat) <- major_topics

major_mat2 <- matrix(0,nrow = N, ncol=Nt)
colnames(major_mat2) <- major_topics

major_mat3 <- matrix(0,nrow = N, ncol=Nt)
colnames(major_mat3) <- major_topics

major_mat4 <- matrix(0,nrow = N, ncol=Nt)
colnames(major_mat4) <- major_topics


# Various methods/prompting
#
# Method V1 => predMacroV1
#   *) for each major asks ChatGPT to choose among the corrspoding minors
#      or 99 = none of the above [minor_mat]
#   *) aggregates all minors and associates them to their majors  [major_mat]
#   *) ask ChatGPT to select one of the majors [major_mat2]
#
# Method V2 => predMacroV2
#   *) picks the selected minors and associates majors and asks ChatGPT
#      to choose form these labels 'major : minor'
#   *) from the selected label 'major : minor' picks the major [major_mat2]
#
# Method => predMacroV3:
#   *) direct classification on all the minors [major_mat3]
#
# Method => predMacroV4:
#   *) direct classification on all the majors [major_mat4]

# select the method you want to test
# you can run them all

method1 <- FALSE
method2 <- TRUE
method3 <- FALSE

useGPT4 <- TRUE

for(i in 1:N){
  try( {
  cat(sprintf("\nText: i=%d\n", i))
  
  if(method1){
    for(lab in major_topics){  # 1) for-loop on the majors
      cat(".")
      mysubtopics <- codebook$subtopic_label[(codebook$major_topic_label %in% lab)]
      nsubt <- length(mysubtopics)
      myhint <- paste0("\n\nAnswer briefly using only the category numbers below, or NA:\n",
                       paste0(sprintf("%d = %s\n", 1:nsubt, mysubtopics),collapse = ""), collapse="")
      myhint <- paste0(myhint, "Category:99 = none of the above", collapse = "")
      myhint <- paste0(myhint, "Category number: ", collapse = "")
      
      classified <- NULL
      while(is.null(classified)){
        ans <- try(askGPT(what, dat$text[i], myhint, useGPT4, TRUE), TRUE)
        classified <- ans$choices[[1]]$message$content
        if(!is.null(classified)){
          pattern <- "\\d+" # Match one or more digits
          matches <- as.integer(unlist(str_extract_all(classified, pattern)))
          minor_mat[i, lab] <- minor_topics_num[match(mysubtopics[matches], minor_topics)][1]
        }
      }
    } # end for-loop on the majors. 
    # At this point, for each major there is either a 99 or the selected one.
    # The matrix minor_mat has as many columns as the majors.
    # Every row contains either a 99 or the minor selected

    idx <- which(!is.na(minor_mat[i,]))
    if(length(idx)>0){  # selection of all minors from minor_mat 
                        # asks ChatGPT to choose among "major : minor"
      mytopics <- colnames(minor_mat)[idx]
      ntop <- length(mytopics)
      myhint2 <- paste0("\n\nAnswer briefly using only the category numbers below, or NA:\n",
                        paste0(sprintf("%d = %s\n", 1:ntop, mytopics),collapse = ""), collapse="")
      myhint2 <- paste0(myhint2, "Category number: ", collapse = "")
      
      classified2 <- NULL
      cat("\n")
      while(is.null(classified2)){
        ans <- try(askGPT(what, dat$text[i], myhint2, useGPT4, TRUE), TRUE)
        classified2 <- ans$choices[[1]]$message$content
        pattern <- "\\d+" # Match one or more digits
        matches <- as.integer(unlist(str_extract_all(classified2, pattern)))
        major_mat[i, mytopics[matches]] <- 1  # once the subtopic is selcted, its major is associated
      }
    }
    
    subId <- minor_mat[i,][which(!is.na(minor_mat[i,]))]
    subId <- subId[subId>0]
    if(length(subId)>0){
      
      mystopics <- minor_topics[match(subId, minor_topics_num)]
      nstop <- length(mystopics)
      myhint3 <- paste0("\n\nAnswer briefly using only the category numbers below, or NA:\n",
                        paste0(sprintf("%d = %s: %s\n", 1:nstop, names(subId), mystopics),collapse = ""), collapse="")
      myhint3 <- paste0(myhint3, "Category number: ", collapse = "")
       
      classified3 <- NULL
      cat("\n")
      while(is.null(classified3)){
        ans <- try(askGPT(what, dat$text[i], myhint3, FALSE, TRUE), TRUE)
        classified3 <- ans$choices[[1]]$message$content
        pattern <- "\\d+" # Match one or more digits
        matches <- as.integer(unlist(str_extract_all(classified3, pattern)))
        final <- codebook$major_topic_label[match(mystopics[matches], codebook$subtopic_label)]
        major_mat2[i, final] <- 1
      }
    }
    
    
    
    
  }
  
  if(method2){ # full labels of subtopics
    mysublabtopics <- minor_topics
    nsublabt <- length(mysublabtopics)
    myhint4 <- paste0("\n\nAnswer briefly using only the category numbers below:\n",
                      paste0(sprintf("%d = %s\n", 1:nsublabt, mysublabtopics),collapse = ""), collapse="")
    myhint4 <- paste0(myhint4, "Category number: ", collapse = "")
    
    classified4 <- NULL
    while(is.null(classified4)){
      ans4 <- try(askGPT(what, dat$text[i], myhint4, useGPT4, TRUE, temp=0.5), TRUE)
      classified4 <- ans4$choices[[1]]$message$content
      if(!is.null(classified4)){
        pattern <- "\\d+" # Match one or more digits
        matches <- as.integer(unlist(str_extract_all(classified4, pattern)))
        labMinor <- mysublabtopics[matches]
        labMajor <- codebook$major_topic_label[match(labMinor, codebook$subtopic_label)]
        major_mat3[i, labMajor]  <- codebook$subtopic[match(labMinor, codebook$subtopic_label)]
      }
    }
  }
    
  if(method3){
    mylabtopics <- major_topics
    nlabt <- length(mylabtopics)
    myhint5 <- paste0("\n\nAnswer briefly using only the category numbers below:\n",
                        paste0(sprintf("%d = %s\n", 1:nlabt, mylabtopics),collapse = ""), collapse="")
      myhint5 <- paste0(myhint5, "Category number: ", collapse = "")
      
      classified5 <- NULL
      while(is.null(classified5)){
        ans5 <- try(askGPT(what, dat$text[i], myhint5, useGPT4, TRUE, temp=0.5), TRUE)
        classified5 <- ans5$choices[[1]]$message$content
        if(!is.null(classified5)){
          pattern <- "\\d+" # Match one or more digits
          matches <- as.integer(unlist(str_extract_all(classified5, pattern)))
          lab <- codebook$major_topic_label[match(major_topics[matches], codebook$major_topic_label)]
          major_mat4[i, lab] <- major_topics[matches]
        }
      }
    }
    
  Sys.sleep(1)
  }, TRUE)
}

predMacroV1 <- unlist(apply(major_mat,1,function(x) colnames(major_mat)[which(x>0)][1]))
predMacroV2 <- unlist(apply(major_mat2,1,function(x) colnames(major_mat2)[which(x>0)][1]))
predMacroV3 <- unlist(apply(major_mat3,1,function(x) colnames(major_mat3)[which(x>0)][1]))
predMacroV4 <- unlist(apply(major_mat4,1,function(x) colnames(major_mat4)[which(x>0)][1]))

minorCode <- NULL
minorLabel <- NULL
for(i in 1:nrow(minor_mat)){
  minorCode <- c(minorCode, minor_mat[i,match(predMacroV2[i], colnames(minor_mat))])
  minorLabel <- c(minorLabel, codebook$subtopic_label[match(minorCode[i], codebook$subtopic)])
}

minorCodeV3 <- NULL
minorLabelV3 <- NULL
for(i in 1:nrow(major_mat3)){
  minorCodeV3 <- c(minorCodeV3, major_mat3[i,match(predMacroV3[i], colnames(major_mat3))])
  minorLabelV3 <- c(minorLabelV3, codebook$subtopic_label[match(minorCodeV3[i], codebook$subtopic)])
}


tab <- data.frame(idCoding = dat$id.coding[1:N],
                  idRandom = dat$id.random.stratif[1:N], # aggiunto da Marcello con training set 4 e 5 per identificare più facilmente le obs
                  idOrig = dat$id[1:N],
                  text = dat$text[1:N],
           majorV1 = predMacroV1,
           majorV1Code = codebook$major[match(predMacroV1, codebook$major_topic_label)], 
           minorCode = minorCode,
           minorLabel = minorLabel,
           majorV2 = predMacroV2,
           majorV2Code = codebook$major[match(predMacroV2, codebook$major_topic_label)],
           majorV3 = predMacroV3,
           minorCodeV3 = minorCodeV3,
           minorLabelV3 = minorLabelV3,
           majorV3Code = codebook$major[match(predMacroV3, codebook$major_topic_label)],
           majorV4 = predMacroV4,
           majorV4Code = codebook$major[match(predMacroV4, codebook$major_topic_label)],
           GPT4 = useGPT4,
           stringsAsFactors = FALSE)
rownames(tab) <- tab$idCoding


write.csv(tab,file="training_set_coded.csv", row.names = FALSE)
write_xlsx(tab,path="training_set_coded.xlsx")


