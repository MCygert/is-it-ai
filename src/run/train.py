import gc
import pickle
from dataclasses import dataclass
from typing import Tuple

from datasets import Dataset
from lightgbm import LGBMClassifier
from pandas import DataFrame
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerFast

from utils.data_ops import get_whole_dataset


@dataclass
class CONFIG:
    lowercase: bool = False
    vocab_size: int = 30522
    path_to_dumping_model: str = "models/checkpoints/ensemble.pkl"


def dummy(text):
    return text


def get_data() -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    root_path = "data/"
    full_training_data = get_whole_dataset(root_path)
    X = full_training_data["text"]
    y = full_training_data["label"]
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True
    )
    return x_train, x_test, y_train, y_test


def get_model() -> VotingClassifier:
    clf = MultinomialNB(alpha=0.1)
    sgd_model = SGDClassifier(max_iter=8000, tol=1e-4, loss="modified_huber")
    p6 = {
        "n_iter": 2500,
        "verbose": -1,
        "objective": "cross_entropy",
        "metric": "auc",
        "learning_rate": 0.01,
        "colsample_bytree": 0.78,
        "colsample_bynode": 0.8,
        "lambda_l1": 4.562963348932286,
        "lambda_l2": 2.97485,
        "min_data_in_leaf": 115,
        "max_depth": 23,
        "max_bin": 898,
    }

    lgb = LGBMClassifier(**p6)
    weights = [0.068, 0.311, 0.31]

    ensemble = VotingClassifier(
        estimators=[("mnb", clf), ("sgd", sgd_model), ("lgb", lgb)],
        weights=weights,
        voting="soft",
        n_jobs=-1,
    )
    return ensemble


if __name__ == "__main__":
    x_train, x_test, y_train, y_test = get_data()
    vocab = CONFIG.vocab_size

    raw_tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    raw_tokenizer.normalizer = normalizers.Sequence(
        [normalizers.NFC()] + [normalizers.Lowercase()] if CONFIG.lowercase else []
    )
    raw_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=special_tokens)
    dataset = Dataset.from_dict({"text": x_test, "label": y_test})

    def train_corp_iter():
        for i in range(0, len(dataset), 1000):
            yield dataset[i : i + 1000]["text"]

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

    vectorizer = TfidfVectorizer(
        ngram_range=(3, 5),
        lowercase=False,
        sublinear_tf=True,
        vocabulary=vocab,
        analyzer="word",
        tokenizer=dummy,
        preprocessor=dummy,
        token_pattern=None,
        strip_accents="unicode",
    )

    tf_train = vectorizer.fit_transform(tokenized_texts_train)
    tf_test = vectorizer.transform(tokenized_texts_test)

    del vectorizer
    gc.collect()
    ensemble = get_model()
    ensemble.fit(tf_train, y_train)
    gc.collect()
    with open(CONFIG.path_to_dumping_model, "wb") as f:
        pickle.dump(ensemble, f)
