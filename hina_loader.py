# -*- coding: utf-8 -*-
""" Loads the dataset from the provided xml format, and returns
articles as named tuples which consist of article text, and 
the topic and genre labels
"""

from collections import namedtuple
from xml.etree import ElementTree as ET

TAG_PREFIX_1 = "{http://iptc.org/std/nar/2006-10-01/}"
TAG_PREFIX_2 = "{http://www.w3.org/1999/xhtml}"

IGNORE_TOPICS = [
    "Hrvatska",
    "Izbori2009",
    "LZMK",
    "Ličnosti",
    "MP2009",
    "Svijet",
    "VladaRH",
]

# text, genre -> str
# topics -> list of strings
Article = namedtuple("Article", ["text", "topics", "genre"])


def load_full_hina_dataset(path):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    path : str
             The path to the dataset in the expected xml format

    Returns
    -------
    articles : list(namedtuple)
            The list of parsed articles, each consists of
            the article text, and the topic and genre labels

    """

    root_node = ET.parse(path, parser=ET.XMLParser(encoding="utf-8")).getroot()
    articles = []

    for newsitem_node in root_node:
        contentMeta_node = newsitem_node.find(TAG_PREFIX_1 + "contentMeta")

        if contentMeta_node is not None:
            topics = []
            # find all topics
            for subject_node in contentMeta_node.findall(TAG_PREFIX_1 + "subject"):
                if subject_node.get("type") != "hibtype:theme":
                    # we only want topics
                    continue

                theme_str = subject_node.find(TAG_PREFIX_1 + "name").text
                if theme_str is not None:
                    theme_str = theme_str.strip()
                if theme_str not in IGNORE_TOPICS:
                    topics.append(theme_str)

            # find genre
            genre_nodes = contentMeta_node.findall(TAG_PREFIX_1 + "genre")
            genre_str = genre_nodes[0].find(TAG_PREFIX_1 + "name").text
            if genre_str is not None:
                genre_str = genre_str.strip()

        else:
            topics = None
            genre_str = None

        # get article body
        node = newsitem_node
        node = node.find(TAG_PREFIX_1 + "contentSet")
        node = node.find(TAG_PREFIX_1 + "inlineXML")

        if node is not None:
            node = node.find(TAG_PREFIX_2 + "html")
            node = node.find(TAG_PREFIX_2 + "body")
            text = node.text
            if text is not None:
                text = text.strip()

        else:
            text = None

        if text is not None:
            article = Article(text=text, topics=topics, genre=genre_str)
            articles.append(article)

    return articles
