import re
import string

ESCAPE_PUNCT_RE = re.compile('[%s]' % re.escape(string.punctuation))
COMMON_REPLACEMENTS = [('w/', 'with'),
                       ('b/', 'between'),
                       ('&', 'and'),
                       ("'cause", 'because'),
                       ('and/or', 'and - or'),
                       ("'an", 'and'),
                       ("'n", 'and'),
                       ('mos', 'months'),
                       ('sec', 'second'),
                       ("@", 'at')]

# TODO:
# Entry Type Entry Example Recommended Replacement:
# E-mail Address your.name@example.com subEmailaddress
# URL address http://www.LIWC.net subURLaddress
# Hashtag #LIWCisawesome subHashtag
# Twitter Handle @jwpennebaker subTwittername
def substitutions(x):
    for orig, repl in COMMON_REPLACEMENTS:
        x=x.replace(orig, repl)
    return x

def remove_urls(x):
    return re.sub("http(.+)?(\W|$)", ' ', x)


def normalize_spaces(x):
    return re.sub("[\n\r\t ]+", ' ', x)


def escape_punct(x):
    return ESCAPE_PUNCT_RE.sub(' ', x)


def lower(x):
    return x.lower()


def preprocess_pre_tokenizing(x):
    return substitutions(
        normalize_spaces(
            remove_urls(x)))


def preprocess_e2e(x):
    return escape_punct(
        lower(
            preprocess_pre_tokenizing
            (x)))
