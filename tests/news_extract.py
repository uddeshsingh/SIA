import os
import sys
import requests
import json
from time import sleep
from datetime import datetime

import pandas as pd
from rich.console import Console
from newsplease import NewsPlease

console = Console()


class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super(DateTimeEncoder, self).default(obj)
    
def main():
    df = pd.read_csv("top10s.csv")

    checkpoint = { element: 0 for element in df['Name']}
    if os.path.exists("checkpoint_parsing"):
        with open("checkpoint_parsing", "r") as file:
            checkpoint = json.load(file)
    with console.status("[bold green] Extracting news from HTML") as status:
        for company in df['Name']:
            os.makedirs(f"News/BingNewsText/{company}/", exist_ok=True)
            console.log(f"[bold] {company} crawling")
        # Call the API
            htmls = [ f"News/BingNewsHtml/{company}/{file}" for file in os.listdir(f"News/BingNewsHtml/{company}")]
            for i, html in enumerate(htmls, 1):
                if (i-1) < checkpoint[company]:
                    continue
                fileName = f"News/BingNewsText/{company}/{os.path.basename(html).split(sep='.', maxsplit=2)[0]}.txt"
                try:
                    article = dict()
                    with open(html, 'r') as htmlFile:
                        status.update(f"[bold green] News Extraction [/bold green] [red] parsing [/red] ({html}.../{i/len(htmls):.2%})")
                        article = NewsPlease.from_html("".join(htmlFile.readlines()))
                    
                    if type(article) is dict or article.maintext is None:
                        continue
                    with open(fileName, 'w') as textFile:
                        status.update(f"[bold green] News Extraction [/bold green] [red] saving [/red] ({article.title[:50]}.../{i/len(htmls):.2%})")
                        json.dump(article.get_serializable_dict(), textFile, indent=4)
                    
                    status.update(f"[bold green] News Extraction [/bold green] [red] updating [/red] ({article.title[:50]}.../{i/len(htmls):.2%})")
                    checkpoint[company] += 1
                    with open("checkpoint_parsing", "w") as file:
                        file.write(json.dumps(checkpoint, indent=4))
                except Exception as e:
                    console.log(f"[bold red] {fileName} save failed ({e})")
            console.log(f"{company} finished")

    return 0


if __name__ == "__main__":
    sys.exit(main())