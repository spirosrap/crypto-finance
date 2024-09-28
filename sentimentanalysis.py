import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#nltk.download('vader_lexicon')
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from coinbaseservice import CoinbaseService
from technicalanalysis import TechnicalAnalysis
from newsapi import NewsApiClient
from config import NEWS_API_KEY

class SentimentAnalysis:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def analyze_sentiment(self, keyword):
        try:
            # Using NewsAPI to fetch recent news articles
            newsapi = NewsApiClient(api_key=NEWS_API_KEY)
            articles = newsapi.get_everything(q=keyword, language='en', sort_by='publishedAt', page_size=100)
            
            # Analyze sentiment for each article
            sentiments = []
            for article in articles['articles']:
                title = article.get('title', '')
                description = article.get('description', '')
                text = f"{title} {description}".strip()
                if text:
                    sentiment = self.sia.polarity_scores(text)
                    sentiments.append(sentiment['compound'])
            
            if not sentiments:
                return "Unable to analyze"
            
            # Calculate average sentiment
            avg_sentiment = sum(sentiments) / len(sentiments)
            
            # Interpret the sentiment
            if avg_sentiment > 0.05:
                return "Positive"
            elif avg_sentiment < -0.05:
                return "Negative"
            else:
                return "Neutral"
        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
            return "Unable to analyze"

    def get_sentiment(self, keyword):
        try:
            # Using NewsAPI to fetch recent news articles
            newsapi = NewsApiClient(api_key=NEWS_API_KEY)
            articles = newsapi.get_everything(q=keyword, language='en', sort_by='publishedAt', page_size=100)
            
            # Analyze sentiment for each article
            sentiments = []
            for article in articles['articles']:
                title = article.get('title', '')
                description = article.get('description', '')
                text = f"{title} {description}".strip()
                if text:
                    sentiment = self.sia.polarity_scores(text)
                    sentiments.append(sentiment['compound'])
            
            if not sentiments:
                return 0  # Neutral sentiment if no articles are found
            
            # Calculate average sentiment
            avg_sentiment = sum(sentiments) / len(sentiments)
            
            # Return the average sentiment score (between -1 and 1)
            return avg_sentiment

        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
            return 0  # Return neutral sentiment in case of error
