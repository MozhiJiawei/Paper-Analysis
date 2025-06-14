"""
Utils包 - 论文分析工具集
提供各种实用工具模块
"""

# 导入子模块
from . import pdf_downloader

__all__ = [
    'pdf_downloader'
]

__version__ = "1.0.0" 


import re
import os.path as op
import urllib
import requests
from bs4 import BeautifulSoup, NavigableString

GROBID_URL = "http://localhost:8070"

def validate_url(path: str):
    """
    Validate a given ``path`` if it is URL or not
    """
    regex = re.compile(
        r"^(?:http|ftp)s?://"  # http:// or https://
        r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|"  # domain...
        r"localhost|"  # localhost...
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
        r"(?::\d+)?"  # optional port
        r"(?:/?|[/?]\S+)$",
        re.IGNORECASE,
    )
    return re.match(regex, path) is not None

def parse_pdf(
    pdf_path: str,
    fulltext: bool = True,
    soup: bool = False,
    return_coordinates: bool = True,
    grobid_url: str = GROBID_URL,
):
    """
    Function to parse PDF to XML or BeautifulSoup using GROBID tool

    You can see http://grobid.readthedocs.io/en/latest/Install-Grobid/ on how to run GROBID locally
    After loading GROBID zip file, you can run GROBID by using the following
    >> ./gradlew run

    Parameters
    ==========
    pdf_path: str or bytes, path or URL to publication or article or bytes string of PDF
    fulltext: bool, option for parsing, if True, parse full text of the article
        if False, parse only header
    grobid_url: str, url to GROBID parser, default at 'http://localhost:8070'
        This could be changed to "https://cloud.science-miner.com/grobid/" for the cloud service
    soup: bool, if True, return BeautifulSoup of the article

    Output
    ======
    parsed_article: if soup is False, return parsed XML in text format,
        else return BeautifulSoup of the XML
    Example
    =======
    >> parsed_article = parse_pdf(pdf_path, fulltext=True, soup=True)
    """
    # GROBID URL
    if fulltext:
        url = "%s/api/processFulltextDocument" % grobid_url
    else:
        url = "%s/api/processHeaderDocument" % grobid_url

    files = []
    if return_coordinates:
        files += [
            ("teiCoordinates", (None, "persName")),
        ]

    if isinstance(pdf_path, str):
        if validate_url(pdf_path) and op.splitext(pdf_path)[-1].lower() != ".pdf":
            print("The input URL has to end with ``.pdf``")
            parsed_article = None
        elif validate_url(pdf_path) and op.splitext(pdf_path)[-1] == ".pdf":
            page = urllib.request.urlopen(pdf_path).read()
            parsed_article = requests.post(url, files={"input": page}).text
        elif op.exists(pdf_path):
            parsed_article = requests.post(
                url, files={"input": open(pdf_path, "rb")}
            ).text
        else:
            parsed_article = None
    elif isinstance(pdf_path, bytes):
        # assume that incoming is byte string
        parsed_article = requests.post(url, files={"input": pdf_path}).text
    else:
        parsed_article = None

    if soup and parsed_article is not None:
        parsed_article = BeautifulSoup(parsed_article, "lxml-xml")
    return parsed_article

def parse_abstract(article):
    """
    Parse abstract from a given BeautifulSoup of an article
    """
    div = article.find("abstract")
    abstract = ""
    for p in list(div.children):
        if not isinstance(p, NavigableString) and len(list(p)) > 0:
            abstract += " ".join(
                [elem.text for elem in p if not isinstance(elem, NavigableString)]
            )
    return abstract

def parse_sections(article, as_list: bool = False):
    """
    Parse list of sections from a given BeautifulSoup of an article

    Parameters
    ==========
    as_list: bool, if True, output text as a list of paragraph instead
        of joining it together as one single text
    """
    article_text = article.find("text")
    divs = article_text.find_all("div", attrs={"xmlns": "http://www.tei-c.org/ns/1.0"})
    sections = []
    for div in divs:
        head_number = ""
        heading = ""
        text = []
        head_tag = div.find('head')
        if head_tag is not None:
            n_value = head_tag.get('n')
            head_text = head_tag.text
            if head_text is not None:
                heading = str(head_text)
            if n_value is not None:
                head_number = str(n_value)
        p_tags = div.find_all('p')
        for p in p_tags:
            if p is not None:
                try:
                    text.append(p.text)
                except:
                    pass

        if heading is not "" or len(text) > 0:
            sections.append(
                {
                    "heading": heading,
                    "head_number": head_number,
                    "text": text,
                }
            )
    return sections

def convert_article_soup_to_dict(article, as_list: bool = False):
    """
    Function to convert BeautifulSoup to JSON format
    similar to the output from https://github.com/allenai/science-parse/

    Parameters
    ==========
    article: BeautifulSoup

    Output
    ======
    article_json: dict, parsed dictionary of a given article in the following format
        {
            'title': ...,
            'abstract': ...,
            'sections': [
                {'heading': ..., 'text': ...},
                {'heading': ..., 'text': ...},
                ...
            ],
            'references': [
                {'title': ..., 'journal': ..., 'year': ..., 'authors': ...},
                {'title': ..., 'journal': ..., 'year': ..., 'authors': ...},
                ...
            ],
            'figures': [
                {'figure_label': ..., 'figure_type': ..., 'figure_id': ..., 'figure_caption': ..., 'figure_data': ...},
                ...
            ]
        }
    """
    article_dict = {}
    if article is not None:
        title = article.find("title", attrs={"type": "main"})
        title = title.text.strip() if title is not None else ""

        article_dict["title"] = title
        article_dict["abstract"] = parse_abstract(article)
        article_dict["sections"] = parse_sections(article, as_list=as_list)

        doi = article.find("idno", attrs={"type": "DOI"})
        doi = doi.text if doi is not None else ""
        article_dict["doi"] = doi

        return article_dict
    else:
        return None

def parse_pdf_to_dict(
    pdf_path: str,
    fulltext: bool = True,
    soup: bool = True,
    as_list: bool = True,
    return_coordinates: bool = True,
    grobid_url: str = GROBID_URL,
):
    """
    Parse the given PDF and return dictionary of the parsed article

    Parameters
    ==========
    pdf_path: str, path to publication or article
    fulltext: bool, whether to extract fulltext or not
    soup: bool, whether to return BeautifulSoup or not
    as_list: bool, whether to return list of sections or not
    grobid_url: str, url to grobid server, default is `GROBID_URL`
        This could be changed to "https://cloud.science-miner.com/grobid/" for the cloud service

    Ouput
    =====
    article_dict: dict, dictionary of an article
    """
    parsed_article = parse_pdf(
        pdf_path,
        fulltext=fulltext,
        soup=soup,
        return_coordinates=return_coordinates,
        grobid_url=grobid_url,
    )
    article_dict = convert_article_soup_to_dict(parsed_article, as_list=as_list)
    return article_dict


if __name__ == "__main__":
    article_dict = parse_pdf_to_dict("/home/ljw2/scipdf/Talaria Interactively Optimizing Machine Learning Models for Efficient Inference.pdf")
    print(article_dict)
    
