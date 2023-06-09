"""Main module."""
import numpy as np
import os.path
import re
import json
from functools import partial

import pandas as pd
from collections import defaultdict


def read_dic(lexicon_path='LIWC2015.dic'):
    indices_to_categories = dict()

    with open(lexicon_path, encoding='utf8') as f:
        contents = f.read()

    _, categories_text, words_text = contents.split('%')
    for cat_line in categories_text.strip().split('\n'):
        idx, cat = cat_line.split('\t')
        indices_to_categories[idx.strip()]=cat.strip()

    categories_to_words = {cat:list() for cat in indices_to_categories.values()}
    for word_line in words_text.strip().split('\n'):
        words = word_line.split('\t')
        word, cats = words[0], list(map(lambda x:indices_to_categories[x.strip()], words[1:]))
        for cat in cats:
            categories_to_words[cat].append(word.lower())
    return categories_to_words

def read_json(lexicon_path='LIWC2015.jsonl'):
    with open(lexicon_path, 'r') as f:
        return {k.lower():v for k, v in json.load(f).items()}

def read_liwc(lexicon_path='LIWC2015.jsonl'):
    _, ext = os.path.splitext(lexicon_path)
    if ext in {'.json', '.jsonl'}:
        return read_json(lexicon_path)
    elif ext =='.dic':
        return read_dic(lexicon_path)
    else:
        raise ValueError(f'Unsupported file extension: {ext}')

def lex_to_regex(lexicon_list):

    # regex = r''
    terms = list()
    for term in lexicon_list:
        term = term.replace("*",".*")
        if '(discrep' in term:
            term = term.replace('(discrep)', '(?P=discrep)')
        elif '(53' in term:
            term = term.replace('(53)', '(?P=53)')
        elif ('(' in term) and (')' in term):
            term = term.replace('(', '(?:')
        else:
            term = term.replace(")",r"[)]")
            term = term.replace("(", r"[(]")
        terms.append(r'\b{}\b'.format(term))
        # regex = regex + r'|\b'+ term + r'\b'
    # regex = regex.removeprefix('|')
    regex=r'|'.join(terms)
    raw_s = r'{}'.format(regex)
    return raw_s

def get_matchers(lexica):
    regexes_dict = dict()
    for lexicon_name, lexicon_list in sorted(lexica.items()):
        the_regex= lex_to_regex(lexicon_list)
        regexes_dict[lexicon_name.lower()] = the_regex
    if '?P=discrep' in regexes_dict['posemo']:
        regexes_dict['posemo']=regexes_dict['posemo'].replace('?P=discrep', regexes_dict['discrep'])
    #TODO: there is a (53) group before a "like"; investigate what it means. is that the index of a LIWC category?
    if '?P=53' in regexes_dict['affect']:
        regexes_dict['affect']=regexes_dict['affect'].replace('?P=53', regexes_dict['reward'])
    regexes =list()
    for lexicon_name, lexicon_re in sorted(regexes_dict.items()):
        the_regex= r'(?P<{}>{})'.format(lexicon_name, lexicon_re)
        regexes.append(the_regex)
    regexes.append(r'(?P<tokens>\b\w+\b)')
    return regexes

def match_sent(sent: str, matchers):
    """
    returns a dictionary where the key is the name of the lexicon, and the value
    a list of the matching strings
    """
    to_return=defaultdict(list)
    for matcher in matchers:
        for match in re.finditer(matcher, sent, flags=re.I):
            for lex_name, matched_word in match.groupdict().items():
                if matched_word is not None:
                    to_return[lex_name].append(matched_word)
    return dict(to_return)

def bag_of_lexicons(sent:str, matchers):
    return {k: len(v) for k, v in match_sent(sent, matchers).items()}
def bag_of_lexicons_as_series(sent:str, matchers):
    return pd.Series(bag_of_lexicons(sent, matchers), dtype=np.int)

def df_liwcifer(df:pd.DataFrame, text_col:str, matchers):
    return df[text_col].apply(partial(bag_of_lexicons_as_series, matchers=matchers)).fillna(0)

if __name__ == '__main__':
    lexicon_path = '../LIWC2015_English_OK.dic'
    lexica = read_liwc(lexicon_path)
    matchers = get_matchers(lexica)
    documents = ['this is a document',
                 'this is another document',
                 'there are so many documents in here']
    print('finding tokens that match with LIWC')
    for sent in documents:
        print(sent, match_sent(sent, matchers))

    print('\nextracting word counts')
    for sent in documents:
        print(sent, bag_of_lexicons(sent, matchers))

    print('\nextracting word counts from a pandas DataFrame')
    document_df = pd.DataFrame({'text':documents},
                               index=['a', 'b', 'c'])
    print(df_liwcifer(document_df,'text', matchers))
