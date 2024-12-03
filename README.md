# SIA
Interactive Stock Investment Advisor (SIA) that provides users with company/stock information through a Q&A format.

## Project Proposal: Stock Investment Advisor (SIA)

### Getting started

1. First, install depedent libraries

```shell
$ pip install -r requirements.txt
```

2. Prepare tokens for Discord bot and Vertex AI (Google) in your shell script!

```shell
$ vi ~/.zshrc
export DISCORD_TOKEN="Paste your bot token!"
export VERTEX_AI_API_KEY="Paste your Vertex AI API key!"
```

3. Run chatbot!

```shell
$ python chatbot.py
```

### Goal

To create an interactive stock investment advisor that provides users with company information through a Q&A format. Our goal is to empower investors and financial enthusiasts with accurate, timely, and insightful information.

### Features

* Access to Financial Statements/Reports of Tech Companies</br>
-> Get instant access to the latest financial reports and statements from tech companies.

* Key Points from Financial Statements/Reports</br>
-> Stay up-to-date on the key performance indicators (KPIs) that matter most for your investments.

* Summarization of the Latest Related News with Sentiment Analysis on the News</br>
-> Receive expertly curated news summaries, complete with sentiment analysis to help you make informed investment decisions.

* Organized Report Detailing Company Performance </br>
-> Get a comprehensive overview of each company's performance, including metrics and insights that matter most for your investments.

* Insights on Whether the Company’s Stock is Growing or Declining</br>
-> Make data-driven decisions with our expertly crafted stock performance insights.

## Scope

Our project will focus on the following key elements:

* **Coverage**: We will cover tech companies in the top 50, providing a comprehensive overview of their financial health and market performance.

* **Time Period for Report Summarization**: We will summarize reports for the current year 2024.

* **Time Period for News Summarization**: We will provide news summaries up to the current date, with a focus on the last 12 months.

* **APIs and News Sources**: Our API integration will be limited to no more than two sources each, ensuring efficient data retrieval and minimal latency.

## Team

Meet our talented team of engineers and researchers:

### Chaeeon Lim

* **News Information Extraction**
    * **Data Collection**: Parsing financial reports, SEC filings, and earnings call transcripts to collect valuable data.
    * **Event Extraction**: Identifying key metrics, events, and insights from text data.
    * **Named Entity Recognition**: Extracting named entities like company names, people, locations, etc.
* **Document Classification**
    * **Topic-specific Feature Engineering**: Sorting news articles by topic (e.g. mergers, earnings, leadership changes) 
    * **Text categorization**: Classifying company filings by type 
    * **Relevance Scoring**: Identifying relevant vs. irrelevant information.
* **Dialogue Assistant**
    * **Financial summary Generation**: Summarizing key financial metrics and performance 
    * **Recommendation Generation**: Generating portfolio reviews and market updates 
    * **Natural Language Generation**: Producing customized client communications 

### Uddesh Santosh Kumar Singh

* **Report Information Extraction**
    * **Named Entity Recognition**: Identifying the various metrics and entities mentioned in Financial Reports.
    * **Information Retrieval**: Find and retrieve structured information from long reports.
    * **Data Mining**: Locate key words and metrics in unstructured data within reports.
* **Sentiment Analysis**
    * **Sentiment Classification**: Classify text into predefined sentiment categories, such as positive, negative, or neutral 
    * **Metric-Based Sentiment Understanding**: Understand the emotions conveyed by the structured data. Example- Missed Target/ On Target/ Exceeded Target for net profits 
* **Dialogue Assistant**
    * **Intent Recognition**: Classify the user's intent, such as retrieving financial data, asking for stock performance, or requesting sentiment analysis of a news event 
    * **Query Expansion and Paraphrasing**: Rephrase user queries to improve the accuracy of results, and refine responses based on user input 
    * **Response Generation**: Generate coherent and informative responses to user queries, using data from financial reports, sentiment analysis, and other sources 

## Data Sources

Our project will leverage the following open-source APIs and news sources:
* Available Open-Source APIs for financial news 
* Investing.com/Companies investor relations website 

### Open-Source APIs
* [Sentiment Analysis](https://www.kaggle.com/code/indermohanbains/news-sentiment-analysis), [public APIs](https://github.com/public-apis/public-apis?tab=readme-ov-file#news)
* News APIs: [Associated Press](https://developer.ap.org/), [New York Times](https://developer.nytimes.com/), [Bloomberg API](https://www.bloomberg.com/professional/support/api-library/)
* Financial APIs: [News api](https://newsapi.org/pricing), [Mediastack](https://mediastack.com/product), [Marketaux](https://www.marketaux.com/pricing)