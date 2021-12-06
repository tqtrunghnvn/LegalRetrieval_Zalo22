import json

def eda_corpus(corpus_path):
    f_corpus = open(corpus_path)
    corpus = json.load(f_corpus)
    # print('corpus: ', type(corpus))
    n_laws = len(corpus)
    print('n_laws: ', n_laws)

    n_articles_per_law = {}
    for law in corpus:
        n_articles = len(law['articles'])
        if n_articles not in n_articles_per_law.keys():
            n_articles_per_law[n_articles] = 1
        else:
            n_articles_per_law[n_articles] += 1
    print('n_articles_per_law: ', n_articles_per_law.keys())

if __name__ == '__main__':
    eda_corpus('data/legal_corpus.json')