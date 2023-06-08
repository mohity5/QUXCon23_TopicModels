from spacy import load as loader
SpacyTokeniser = loader('en_core_web_sm')


from nltk.tokenize.treebank import TreebankWordDetokenizer
d = TreebankWordDetokenizer()

def lemmatize(data):

    Lemmatised_Text = list()
    counter = 0
    doc_len = list()

    for doc in data:

        counter = counter + 1
        if((counter%1000) == 0):
            print(counter,' documents processed.')
        
        doc = doc.lower()
        spacyDoc = SpacyTokeniser(doc)
        sentanceTokens = list()

        for token in spacyDoc:
            if(token.is_stop is False):
                sentanceTokens.append(token.lemma_)

        #This could be done before the stopword comparision, currently for better accuracy.
        if((len(sentanceTokens) > 20) & (len(sentanceTokens) < 10000)) :
            Lemmatised_Text.append(sentanceTokens)
            doc_len.append(len(sentanceTokens))


    Lemmatised_Text = [d.detokenize(doc) for doc in Lemmatised_Text]
    return Lemmatised_Text ,doc_len

                  


