import os
import sys
import requests
import json
from time import sleep
from datetime import datetime

import pandas as pd
from rich.console import Console
from newsplease import NewsPlease, NewsArticle

console = Console()


class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super(DateTimeEncoder, self).default(obj)
    
def main():
    with open("token.json", "r") as f:
        tokens = json.load(f)


    df = pd.read_csv("top10s.csv")

    subscription_key = tokens['bing']
    since = int(datetime(2023, 1, 1).timestamp())
    count = 100
    offset = 0

    # Construct a request
    search_url = "https://api.bing.microsoft.com/v7.0/news/search"
    headersApi = {"Ocp-Apim-Subscription-Key" : subscription_key}
    headersCrawl = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

    checkpoint = { element: 0 for element in df['Name']}
    if os.path.exists("checkpoint"):
        with open("checkpoint", "r") as file:
            checkpoint = json.load(file)
    with console.status("[bold green] crawling Bing News") as status:
        for company in df['Name']:
            os.makedirs(f"News/BingNewsHtml/{company}/", exist_ok=True)
            os.makedirs(f"News/BingNewsJson/", exist_ok=True)
            console.log(f"[bold] {company} crawling")
            cache = []
            params  = {
                "q": company, 
                "since": since,
                "count": count,
                "offset": offset,
                "mkt": "en-US",
                "category": "Busniess",
                "sortBy": "Date",
                "textDecorations": False, "textFormat": "Raw"}
        # Call the API
            try:
                response = requests.get(search_url, headers=headersApi, params=params, timeout=30)
                response.raise_for_status()
                result = response.json()
                with open(f"News/BingNewsJson/{company}_search_result.json", "w") as file:
                    json.dump(response.json(), file, indent=4)

                for i, content in enumerate(result['value'],1):
                    if (i-1) < checkpoint[company]:
                        continue
                    try:
                        status.update(f"[bold green] Bing News [/bold green] [red] crawling [/red] ({content['name'][:50]}.../{i/len(result['value']):.2%})")
                        response = requests.get(content['url'], headers=headersCrawl, timeout=30)
                        status.update(f"[bold green] Bing News [/bold green] [red] saving [/red] ({content['name'][:50]}.../{i/len(result['value']):.2%})")
                        with open(f"News/BingNewsHtml/{company}/{i:03d}_{content['name'].replace(' ', '')[:15]}.html", 'w', encoding='utf-8') as file:
                            file.write(response.text)
                        status.update(f"[bold green] Bing News [/bold green] [red] updating [/red] ({content['name'][:50]}.../{i/len(result['value']):.2%})")
                        checkpoint[company] += 1
                        with open("checkpoint", "w") as file:
                            file.write(json.dumps(checkpoint, indent=4))
                    except Exception as e:
                        console.log(f"[bold] {content['name'][:30]} save failed ({e})")
                console.log(f"bing_{company.lower()}_news.json saved")
            except Exception as ex:
                console.log(ex)

    return 0


if __name__ == "__main__":
    sys.exit(main())