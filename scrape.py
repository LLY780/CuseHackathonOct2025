import trafilatura

def getText(url):
    '''Download and extract text from article url'''
    downloaded = trafilatura.fetch_url(url)
    article_text = trafilatura.extract(downloaded)
    return article_text