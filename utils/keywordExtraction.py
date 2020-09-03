import nltk

class KeywordExtraction():
    def _remove_stopword(self, tokenized_sentence):
        stop_words = set(nltk.corpus.stopwords.words('english'))
        stop_words.update(['.','\'s','\'',',','?','why','how','what','where','when','yes','no'])
        tokenized_sentence = [w.lower() for w in tokenized_sentence if w.lower() not in stop_words]
        return tokenized_sentence

    def _prototype(self, tokenized_sentence):
        lm = nltk.stem.WordNetLemmatizer()
        tokenized_sentence = [lm.lemmatize(w, pos="v") for w in tokenized_sentence]
        return tokenized_sentence

    def get_keyword(self, tokenized_sentence):
        removed_sentence = self._remove_stopword(tokenized_sentence)
        prototype_sentence = self._prototype(removed_sentence)
        prototype_sentence = list(set(prototype_sentence))
        return prototype_sentence