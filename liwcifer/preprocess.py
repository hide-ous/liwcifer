import re
import string

ESCAPE_PUNCT_RE = re.compile('[%s]' % re.escape(string.punctuation))

def remove_urls(x):
    return re.sub("http(.+)?(\W|$)", ' ', x)


def normalize_spaces(x):
    return re.sub("[\n\r\t ]+", ' ', x)


def escape_punct(x):
    return ESCAPE_PUNCT_RE.sub(' ', x)


def lower(x):
    return x.lower()


def preprocess_pre_tokenizing(x):
    return normalize_spaces(
            remove_urls(x))


def preprocess_e2e(x):
    return escape_punct(
        lower(
            preprocess_pre_tokenizing
            (x)))
