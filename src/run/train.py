import os
os.chdir(os.path.expanduser("~/Dev/personal/is_it_ai"))
from utils.data_ops import get_whole_dataset
import gc
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.metrics import roc_auc_score
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

from datasets import Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerFast

from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
import optuna
import gc


root_path = "data/"
full_training_data = get_whole_dataset(root_path)
X = full_training_data["text"]
y = full_training_data['label']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
LOWERCASE = False
VOCAB_SIZE = 30522
raw_tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
raw_tokenizer.normalizer = normalizers.Sequence([normalizers.NFC()] + [normalizers.Lowercase()] if LOWERCASE else [])
raw_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=special_tokens)
dataset = Dataset.from_dict({'text': x_test, 'label': y_test})
def train_corp_iter(): 
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]['text']
raw_tokenizer.train_from_iterator(train_corp_iter(), trainer=trainer)
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=raw_tokenizer,
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
)
tokenized_texts_test = []

for text in tqdm(x_test):
    tokenized_texts_test.append(tokenizer.tokenize(text))

tokenized_texts_train = [] 
for text in tqdm(x_train):
    tokenized_texts_train.append(tokenizer.tokenize(text))
def dummy(text):
    return text
vectorizer = TfidfVectorizer(ngram_range=(3, 5), lowercase=False, sublinear_tf=True, analyzer = 'word',
    tokenizer = dummy,
    preprocessor = dummy,
    token_pattern = None, strip_accents='unicode')

vectorizer.fit(tokenized_texts_test)
vocab = vectorizer.vocabulary_
vectorizer = TfidfVectorizer(ngram_range=(3, 5), lowercase=False, sublinear_tf=True, vocabulary=vocab,
                            analyzer = 'word',
                            tokenizer = dummy,
                            preprocessor = dummy,
                            token_pattern = None, strip_accents='unicode'
                            )

tf_train = vectorizer.fit_transform(tokenized_texts_train)
tf_test = vectorizer.transform(tokenized_texts_test)

del vectorizer
gc.collect()
clf = MultinomialNB(alpha=0.1)
sgd_model = SGDClassifier(max_iter=8000, tol=1e-4, loss="modified_huber") 
p6={'n_iter': 2500,
    'verbose': -1,
    'objective': 'cross_entropy',
    'metric': 'auc',
    'learning_rate': 0.01, 
    'colsample_bytree': 0.78,
    'colsample_bynode': 0.8, 
    'lambda_l1': 4.562963348932286, 
    'lambda_l2': 2.97485, 
    'min_data_in_leaf': 115, 
    'max_depth': 23, 
    'max_bin': 898}

lgb=LGBMClassifier(**p6)
weights = [0.068,0.311,0.31]

ensemble = VotingClassifier(estimators=[('mnb',clf),
                                        ('sgd', sgd_model),
                                        ('lgb',lgb)
                                        ],
                            weights=weights, voting='soft', n_jobs=-1)
ensemble.fit(tf_train, y_train)
gc.collect()
final_preds = ensemble.predict_proba(tf_test)[:,1]