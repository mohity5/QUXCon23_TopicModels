
preProcess <- function(data){
    start <- Sys.time()
    toks <- tokens(data,
                remove_punct = TRUE,
                remove_numbers = TRUE) #,split_hyphens = TRUE


    toks <- tokens_select(toks, min_nchar = 3 , selection = 'remove')
    toks <- tokens_remove(toks, stopwords(language = 'en', source = 'stopwords-iso'))
    toks <- tokens_select(toks, '--+', valuetype = 'regex' , selection = 'remove')



    toks_col <- tokens_select(toks) %>% textstat_collocations(min_count = round(length(data) * 0.00729))
    toks <- tokens_compound(toks, pattern = toks_col)

    stop <- Sys.time()

    print(stop - start)

    return (toks)

}