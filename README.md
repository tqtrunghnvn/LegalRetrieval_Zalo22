pip install googletrans==4.0.0-rc1

# For phoBERT
pip3 install fairseq
pip3 install fastbpe
pip3 install vncorenlp
pip3 install transformers

# Download pretrain phoBERT
https://phamdinhkhanh.github.io/2020/06/04/PhoBERT_Fairseq.html
## Base model
wget https://public.vinai.io/PhoBERT_base_fairseq.tar.gz
tar -xzvf PhoBERT_base_fairseq.tar.gz

## Large model
wget https://public.vinai.io/PhoBERT_large_fairseq.tar.gz
tar -xzvf PhoBERT_large_fairseq.tar.gz

# Library for separating words
mkdir -p vncorenlp/models/wordsegmenter
wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.1.1.jar
wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab
wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr
mv VnCoreNLP-1.1.1.jar vncorenlp/
mv vi-vocab vncorenlp/models/wordsegmenter/
mv wordsegmenter.rdr vncorenlp/models/wordsegmenter/
