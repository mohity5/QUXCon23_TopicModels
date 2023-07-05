
zeroRowIndex <- function(ZeroRowMatrix){
    rowIndexes <- c()

    temp <- strsplit(docnames(ZeroRowMatrix),'text')
    for(row in temp){
        row <- unlist(row)
       rowIndexes <- c(rowIndexes,as.numeric(row[2]))
    }
    return (rowIndexes)

    
}

