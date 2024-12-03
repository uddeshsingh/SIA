import os
import sys
import json
from datetime import datetime

import spacy
import pandas as pd
from rich.console import Console
from newsplease import NewsPlease

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words


console = Console()
nlp = spacy.load('en_core_web_sm')
LANGUAGE = "english"
SENTENCES_COUNT = 5

def summarize(text):
    parser = PlaintextParser.from_string(text, Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)
    summarizer = LsaSummarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)
    
    summary = summarizer(parser.document, SENTENCES_COUNT)
    return ' '.join([str(sentence) for sentence in summary])

# Function to perform Named Entity Recognition
def named_entity_recognition(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Function to extract financial metrics/events
def extract_financial_events(text):
    # Basic rule-based keyword extraction (for demonstration purposes)
    keywords = ['revenue', 'profit', 'acquisition', 'merger', 'earnings']
    events = [line for line in text.split('. ') if any(keyword in line.lower() for keyword in keywords)]
    return events
    
def main():
    df = pd.read_csv("top10s.csv")

    with console.status("[bold green] Extracting Named Entity from News") as status:
        for company in df.iloc:
            ticker = company['Ticker']
            name = company['Name']

            if "Tesla" != name:
                continue
            os.makedirs(f"Results/{ticker}/news", exist_ok=True)
            console.log(f"[bold] {name} crawling")
        # Call the API
            news = []
            newsFiles = [ f"News/BingNewsText/{name}/{file}" for file in os.listdir(f"News/BingNewsText/{name}")]
            for i, newsFile in enumerate(newsFiles, 1):
                try:
                    with open(newsFile, "r") as file:
                        data = json.load(file)
                        if data['maintext'] is  None:
                            continue
                        data['summary'] = summarize(data['maintext'])
                        data['financial_events'] = extract_financial_events(data['maintext'])
                        # data['entities'] = named_entity_recognition(data['maintext'])
                        del data['maintext']
                    
                    status.update(f"[bold green] News Extraction [/bold green] [red] updating [/red] ({data['title'][:50]}.../{i/len(newsFiles):.2%})")
                    with open(f"Results/{ticker}/news/{data['title']}.txt", "w") as file:
                        for k, v in data.items():
                            file.write(f"{k.title()}: {v}\n")
                    # news.append(data)
                except Exception as e:
                    console.log(f"[bold red] {name} news summary save failed ({e})")
            console.log(f"{company} news summary finished")

    return 0


if __name__ == "__main__":
    sys.exit(main())