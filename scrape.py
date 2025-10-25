import trafilatura
# Download and extract main text
def getText(x):
    url = x
    downloaded = trafilatura.fetch_url(url)
    article_text = trafilatura.extract(downloaded)
    return article_text
