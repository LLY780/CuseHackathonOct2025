import trafilatura

url = "https://www.cnn.com/2025/10/25/business/trump-tariffs-canada-reagan"

# Download and extract main text
downloaded = trafilatura.fetch_url(url)
article_text = trafilatura.extract(downloaded)


print(article_text)