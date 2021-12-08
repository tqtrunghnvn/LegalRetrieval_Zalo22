from vncorenlp import VnCoreNLP
rdrsegmenter = VnCoreNLP("pretrain_models/vncorenlp/VnCoreNLP-1.1.1.jar", 
                    annotators="wseg", max_heap_size='-Xmx500m')
from tqdm import tqdm

# Load the model in fairseq
from fairseq.models.roberta import RobertaModel
# phoBERT = RobertaModel.from_pretrained('PhoBERT_base_fairseq', checkpoint_file='model.pt')
phoBERT = RobertaModel.from_pretrained('pretrain_models/PhoBERT_large_fairseq', checkpoint_file='model.pt')
phoBERT.eval()  # disable dropout (or leave in train mode to finetune

from fairseq.data.encoders.fastbpe import fastBPE  
from fairseq import options  
import numpy as np
import random

# Initialize Byte Pair Encoding for PhoBERT
class BPE():
    # bpe_codes = 'PhoBERT_base_fairseq/bpe.codes'
    bpe_codes = 'pretrain_models/PhoBERT_large_fairseq/bpe.codes'
args = BPE()
phoBERT.bpe = fastBPE(args) #Incorporate the BPE encoder into PhoBERT

def fill_mask_sentence(text, n_loops_max=10):
#     print('     original sentence: ', text)
    words_ = rdrsegmenter.tokenize(text)[0]
    n_loops = 0
    n_phoBERT_errros = 0
    # while True:
    while n_loops <= n_loops_max:
        if n_phoBERT_errros > n_loops_max:
            print('phoBERT exceeds n_loops_max: ', n_loops_max)
            break
        words = words_.copy()
        words[random.randint(0, len(words)-1)] = ' <mask>'
        masked_text_tok = ' '.join(words)
        try:
            topk_filled_outputs = phoBERT.fill_mask(masked_text_tok, topk=1)
        except:
            print('     masked_text_tok: ', masked_text_tok)
            # raise
            # sometimes error here
            n_phoBERT_errros += 1
            continue
        new_text = topk_filled_outputs[0][0]
        new_text = new_text.replace('_', ' ')
        new_text = new_text.replace(' ;', ';')
        new_text = new_text.replace(' .', '.')
        new_text = new_text.replace(' :', ':')
        new_text = new_text.replace(' ,', ',')
        new_text = new_text.replace('( ', '(')
        new_text = new_text.replace(' )', ')')
        new_text = new_text.replace(' ?', '?')
        if new_text != text:
#             print('     masked_text_tok: ', masked_text_tok)
#             print('     new text: ', new_text)
            return new_text
        n_loops += 1
        del words
    print('exceed n_loops_max: ', n_loops_max)
    return new_text

def fill_mask_para(text, n_max_random_sentences=10):
#     print('=== original para: ', text)
    sentences = text.rsplit('\n')
    if len(sentences) == 1:
        n_random_sentences = 1
    elif n_max_random_sentences < len(sentences):
        n_random_sentences = random.randint(1, n_max_random_sentences-1)
    else:
        n_random_sentences = random.randint(1, len(sentences)-1)
#     print('n_sentences: ', len(sentences))
#     print('n_random_sentences: ', n_random_sentences)
    random_ids = random.sample(list(range(0, len(sentences))), n_random_sentences)
    for random_id in random_ids:
        new_sentence = fill_mask_sentence(sentences[random_id])
        sentences[random_id] = new_sentence
    new_para = '\n'.join(sentences)
#     print('=== new para: ', new_para)
    return new_para

import os
import json

def gen_corpus():
    corpus_path = '/home/hana/sonnh/zalo-ai-2021/zac2021-ltr-data/new_corpus.json'
    output_path = 'process_corpus'

    with open(corpus_path, 'r') as f_corpus:
        corpus = json.load(f_corpus)
    f_corpus.close()
    n_laws = len(corpus)
    print('n_laws: ', n_laws)

    new_corpus = []
    for law in tqdm(corpus):
        if law['law_id'].find('nđ-cp') != -1:
            new_corpus.append(law)
            continue
        for idx, article in enumerate(law['articles']):
            text = article['text']
            law['articles'][idx]['article_id'] += '_syn'
            if text == '':
                continue
    #         print('=== original text: ', text)
            new_text = fill_mask_para(text, 5)
            # law['articles'][idx]['article_id'] += '_syn'
            law['articles'][idx]['text'] = new_text
    #         print('=== new text: ', new_text)
        new_corpus.append(law)

    print('n_new_laws: ', len(new_corpus))

    out_corpus_file = '/home/hana/sonnh/zalo-ai-2021/zac2021-ltr-data/syn_aug/syn_corpus.json'
    with open(out_corpus_file, 'w', encoding='utf-8') as f_out:
        json.dump(new_corpus, f_out, ensure_ascii=False)
    f_out.close()


def gen_question():
    question_path = '/home/hana/sonnh/zalo-ai-2021/zac2021-ltr-data/train_question_answer.json'
    with open(question_path, 'r') as f_ques:
        ques = json.load(f_ques)
    f_ques.close()
    questions = ques['items']
    n_questions = len(questions)
    print('n_questions: ', n_questions)

    new_questions = []
    for question in tqdm(questions):
        question['question_id'] += '_syn'
        question['question'] = fill_mask_para(question['question'], 5)
        for idx, relevant_article in enumerate(question['relevant_articles']):
            if relevant_article['law_id'].find('nđ-cp') != -1:
                continue
            question['relevant_articles'][idx]['article_id'] += '_syn'
        new_questions.append(question)
    ques['items'] = new_questions

    out_question_file = '/home/hana/sonnh/zalo-ai-2021/zac2021-ltr-data/syn_aug/syn_training_question.json'
    with open(out_question_file, 'w', encoding='utf-8') as f_out:
        json.dump(ques, f_out, ensure_ascii=False)
    f_out.close()


if __name__ == '__main__':
    gen_corpus()
    # gen_question()