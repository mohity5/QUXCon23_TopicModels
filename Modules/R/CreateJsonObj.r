jsonPrep <- function(modl, dfmm){

    phi <- exp(modl@beta) #Topic over terms
    theta <- modl@gamma #Doc over topics
    vocab <- modl@terms

    termFreq <- unlist(colSums(dfmm), use.names = FALSE)
    
    docL <- unlist(rowSums(dfmm), use.names = FALSE)
    docL <- docL[!(docL %in% 0)]

    json <- createJSON(phi,theta,docL,vocab,termFreq)

    return(json)

}