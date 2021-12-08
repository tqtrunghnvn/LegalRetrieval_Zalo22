from vncorenlp import VnCoreNLP
rdrsegmenter = VnCoreNLP("pretrain_models/vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')

text = 'Công an xã xử phạt lỗi không mang bằng lái xe có đúng không?'
text_masked = 'xử_phạt'
words = rdrsegmenter.tokenize(text)[0]
print('words: ', words)
for i, token in enumerate(words):
    if token == text_masked:
        words[i] = ' <mask>'
text_masked_tok = ' '.join(words)
print('text_masked_tok: \n', text_masked_tok)



# Load the model in fairseq
from fairseq.models.roberta import RobertaModel
# phoBERT = RobertaModel.from_pretrained('PhoBERT_base_fairseq', checkpoint_file='model.pt')
phoBERT = RobertaModel.from_pretrained('pretrain_models/PhoBERT_large_fairseq', checkpoint_file='model.pt')
phoBERT.eval()  # disable dropout (or leave in train mode to finetune



from fairseq.data.encoders.fastbpe import fastBPE  
from fairseq import options  
import numpy as np

# Initialize Byte Pair Encoding for PhoBERT
class BPE():
    # bpe_codes = 'PhoBERT_base_fairseq/bpe.codes'
    bpe_codes = 'pretrain_models/PhoBERT_large_fairseq/bpe.codes'
args = BPE()
phoBERT.bpe = fastBPE(args) #Incorporate the BPE encoder into PhoBERT

# Filling marks  
topk_filled_outputs = phoBERT.fill_mask(text_masked_tok, topk=10) 
topk_probs = [item[1] for item in topk_filled_outputs]
print('Total probability: ', np.sum(topk_probs))
print('Input sequence: ', text_masked_tok)
print('Top 10 in mask: ')
for i, output in enumerate(topk_filled_outputs): 
    # print(output[0])
    print(output[0].replace('_', ' '))