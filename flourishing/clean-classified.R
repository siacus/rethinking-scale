# Performs the cleaning of the classifications as models may sometimes answer
# not as required, especially non fine-tuned models
# This script assumes that the object 'classified' exists in the R workspace
# This script should be called from within verify-classification.R
# or verify-test.R
# (S.M.Iacus 2024)

classified$dimension <- tolower(classified$dimension)
classified$value <- tolower(classified$value)

idx <- stringr::str_like(classified$value,"%low%")
classified$value[idx] <-  "low"
idx <- stringr::str_like(classified$value,"%high%")
classified$value[idx] <-  "high"
idx <- stringr::str_like(classified$value,"%medium%")
classified$value[idx] <-  "medium"


classified$dimension <- stringr::str_replace_all(classified$dimension,"-", " ") 
classified$dimension <- stringr::str_replace_all(classified$dimension,"_", " ") 
classified$dimension <- stringr::str_replace_all(classified$dimension,"“", " ") 
classified$dimension <- stringr::str_replace_all(classified$dimension,"'", " ") 
classified$dimension <- stringr::str_replace_all(classified$dimension,"”", " ") 

classified$dimension <- trimws(classified$dimension)





idx <- stringr::str_like(classified$dimension,"%hop%") 
classified$dimension[idx] <-  "having hope"

idx <- stringr::str_like(classified$dimension,"%gratitude%") 
classified$dimension[idx] <-  "expressing gratitude"
idx <- stringr::str_like(classified$dimension,"%grateful%")
classified$dimension[idx] <-  "expressing gratitude"


idx <- stringr::str_like(classified$dimension,"%anxiety%") 
classified$dimension[idx] <-  "anxiety"


idx <- stringr::str_like(classified$dimension,"%lone%") 
classified$dimension[idx] <-  "loneliness"
idx <- stringr::str_like(classified$dimension,"%isol%")
classified$dimension[idx] <-  "loneliness"



idx <- stringr::str_like(classified$dimension,"%peace%") 
classified$dimension[idx] <-   "peace with thoughts and feelings" 

idx <- stringr::str_like(classified$dimension,"%pain%") 
classified$dimension[idx] <-   "feeling pain" 
idx <- stringr::str_like(classified$dimension,"%suffer%")
classified$dimension[idx] <-   "feeling pain" 


idx <- stringr::str_like(classified$dimension,"%gratifi%") 
classified$dimension[idx] <-  "expressing delayed gratification" 

idx <- stringr::str_like(classified$dimension,"%balance%") 
classified$dimension[idx] <-  "balance in the various aspects of own life"

idx <- stringr::str_like(classified$dimension,"%energy%") 
classified$dimension[idx] <-  "having energy"

idx <- stringr::str_like(classified$dimension,"%capability%") 
classified$dimension[idx] <-  "mastery (ability, capability)"

idx <- stringr::str_like(classified$dimension,"%bias%") 
classified$dimension[idx] <-  "perceiving discrimination"
idx <- stringr::str_like(classified$dimension,"%inequity%") 
classified$dimension[idx] <-  "perceiving discrimination"
idx <- stringr::str_like(classified$dimension,"%injustice%") 
classified$dimension[idx] <-  "perceiving discrimination"


idx <- stringr::str_like(classified$dimension,"%death%") 
classified$dimension[idx] <-  "life after death belief"


idx <- stringr::str_like(classified$dimension,"%good%") 
classified$dimension[idx] <-  "promoting good"
idx <- stringr::str_like(classified$dimension,"%promoting%") 
classified$dimension[idx] <-  "promoting good"

idx <- stringr::str_like(classified$dimension,"%esteem%") 
classified$dimension[idx] <-  "self-esteem"


idx <- stringr::str_like(classified$dimension,"%ptsd%") 
classified$dimension[idx] <-  "ptsd (post-traumatic stress disorder)"

idx <- stringr::str_like(classified$dimension,"%satisf%") 
classified$dimension[idx] <-  "life satisfaction"

idx <- stringr::str_like(classified$dimension,"%financial%") 
classified$dimension[idx] <-  "financial/material worry"
idx <- stringr::str_like(classified$dimension,"%material%") 
classified$dimension[idx] <-  "financial/material worry"

idx <- stringr::str_like(classified$dimension,"%belonging%") 
classified$dimension[idx] <-  "belonging to society" 


idx <- stringr::str_like(classified$dimension,"%optimism%") 
classified$dimension[idx] <-  "expressing optimism"
idx <- stringr::str_like(classified$dimension,"%positivity%") 
classified$dimension[idx] <-  "expressing optimism"


idx <- stringr::str_like(classified$dimension,"%energy%") 
classified$dimension[idx] <-  "having energy"
idx <- stringr::str_like(classified$dimension,"%vibes%") 
classified$dimension[idx] <-  "having energy"

idx <- stringr::str_like(classified$dimension,"%fear%") 
classified$dimension[idx] <-  "fear of future"

idx <- stringr::str_like(classified$dimension,"%vital%")
classified$dimension[idx] <-  "vitality"


idx <- stringr::str_like(classified$dimension,"%empathy%")
classified$dimension[idx] <-  "expressing empathy"
idx <- stringr::str_like(classified$dimension,"%empthy%")
classified$dimension[idx] <-  "expressing empathy"
idx <- stringr::str_like(classified$dimension,"%empthy%")
classified$dimension[idx] <-  "expressing empathy"


idx <- classified$dimension %in% c(  "concern for others")
classified$dimension[idx] <-  "expressing empathy"


idx <- stringr::str_like(classified$dimension,"%happi%")
classified$dimension[idx] <-  "happiness"
idx <- stringr::str_like(classified$dimension,"%joy%")
classified$dimension[idx] <-  "happiness"


idx <- classified$dimension %in% c("describing smiling",  
        "expressing positive emotions", 
               "expressing pleasure" , "well being")
classified$dimension[idx] <-  "happiness"


idx <- stringr::str_like(classified$dimension,"%criticism%")
classified$dimension[idx] <-  "religious criticism"


######


  
idx <- classified$dimension %in% c( "feeling loved by a friend" ,"describing social relationships",
"feeling loved by others" )
classified$dimension[idx] <-   "quality of relationships" 

idx <- classified$dimension %in% c("charitable giving/helping others")

idx <- classified$dimension %in% c("belonging to god\\'s love" )
classified$dimension[idx] <-  "feeling loved by god"


idx <- classified$dimension %in% c("describing/drinking related health issues" )
classified$dimension[idx] <-  "describing drinking related health issues"

idx <- classified$dimension %in% c("expressing belief in god", "praying", 
"religious beliefs", "having faith", "describing spiritual beliefs" ,"having faith in god" ,
"having trust in god" , "religious sentiment")
classified$dimension[idx] <-  "belief in god"

idx <- classified$dimension %in% c("feeling government support")
classified$dimension[idx] <-  "expressing government approval"

idx <- classified$dimension %in% c("describing race related health issues", "health limitations",
            "describing health issues" , "mental health", "describing mental health issues")
classified$dimension[idx] <-  "describing health limitations"

idx <- classified$dimension %in% c( "expressing appreciation helping")
classified$dimension[idx] <-  "charitable giving/helping" 

idx <- classified$dimension %in% c( "religious comfort")
classified$dimension[idx] <-  "feeling religious comfort" 

idx <- stringr::str_like(classified$dimension,"%altruism%") 
classified$dimension[idx] <-  "expressing altruism"

idx <- stringr::str_like(classified$dimension,"%political voice%") 
classified$dimension[idx] <-  "feeling having a political voice"


cdims <- sort(unique(classified$dimension))
cdims[which(!(cdims %in% categories))]

residualdims <- classified$dimension[!(classified$dimension %in% categories)]
sort(table(residualdims))


idx  <- which(!(classified$dimension %in% categories))
if(length(idx>0)){
    classified <- classified[-idx,] 
}

idx <- grep("medium", classified$value) # medium or variants
if(length(idx)>0){
    classified$value[idx] <- "medium"  # we assign medium if wrong value
}

idx <- grep("applicable", classified$value) # 'not applicable' or variants
idx <- c(idx, grep("n/a", classified$value) )# n/a or variants
idx <- c(idx, grep("null", classified$value) )# null or variants
idx <- c(idx, grep("na", classified$value) )# NA or variants

if(length(idx)>0){
    classified$value[idx] <- "low"  # we assign low if wrong value
}

# this is mostly for small size parameters like LLAMA-3.2 1B
idx <- c(idx, grep("true", classified$value) )# NA or variants
if(length(idx)>0){
    classified$value[idx] <- "high"  # we assign high if it says 'true'
}
idx <- c(idx, grep("false", classified$value) )# 
if(length(idx)>0){
    classified$value[idx] <- "low"  # we assign low if it says 'false'
}

idx  <- which(!(classified$value %in% c("low","medium","high")))

if(length(idx)>0){
    classified <- classified[-idx,]
}

