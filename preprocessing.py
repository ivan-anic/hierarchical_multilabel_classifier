import re


def remove_nonalpha_and_stopwords(tokenized, stop_words):
    """Removes all non alphabetical characters and stop words from tokens.

    Parameters
    ----------
    tokenized : list(str)
        Tokenized text to be preprocessed

    Returns
    -------
    tuple(str, list(str))
        Raw text and a list of tokens.
    """
    tokens = []
    pattern = re.compile("[\W_]+")
    for token in tokenized:
        pattern.sub("", token)
        if len(token) > 1 and token not in stop_words:
            tokens.append(token)
    return (raw, tokens)
