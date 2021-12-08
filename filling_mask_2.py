from vncorenlp import VnCoreNLP
rdrsegmenter = VnCoreNLP("pretrain_models/vncorenlp/VnCoreNLP-1.1.1.jar", 
                    annotators="wseg", max_heap_size='-Xmx500m')

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


def fill_mask(text, n_sentences=1):    
    words_ = rdrsegmenter.tokenize(text)[0]
    new_texts = []
    remark = ''
    for i in range(0, n_sentences):
        words = words_.copy()
        words[random.randint(0, len(words)-2)] = ' <mask>'
        masked_text_tok = ' '.join(words)
        print('======= {} ========'.format(i))
        print('     masked_text_tok: ', masked_text_tok)
        topk_filled_outputs = phoBERT.fill_mask(masked_text_tok, topk=1)
        new_texts.append(topk_filled_outputs[0][0].replace('_', ' '))
        print('     new text: ', new_texts[-1])
    return new_texts

if __name__ == '__main__':
    text = 'Công an xã xử phạt lỗi không mang bằng lái xe có đúng không?'
    print('Original text: ', text)
    new_texts = fill_mask(text, 3)
    for new_text in new_texts:
        print(new_text)