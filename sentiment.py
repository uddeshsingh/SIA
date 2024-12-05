import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load RoBERTa-based FinBERT model
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Create a sentiment analysis pipeline
finbert_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Ticker symbol mapping
tnc = {
    "apple": "aapl",
    "amazon": "amzn",
    "tesla": "tsla",
    "nvidia": "nvda",
    "salesforce": "crm",
    "broadcom": "avgo",
    "alphabet": "googl",
    "meta": "meta",
    "microsoft": "msft",
    "netflix": "nflx",
    "oracle": "orcl"
}

# Function to split text into chunks
def split_text_into_chunks(text, max_length=512):
    """Splits large text into manageable chunks."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        word_length = len(word) + 1  # Account for space
        if current_length + word_length > max_length:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(word)
        current_length += word_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# Function to aggregate sentiments
def aggregate_sentiments(results, threshold=1):
    """
    Aggregates sentiment scores from all chunks and determines the overall sentiment.
    Ensures NEUTRAL is only returned when it dominates or POSITIVE and NEGATIVE are nearly equal.

    Args:
    - results (list): List of sentiment results with 'label' and 'score'.
    - threshold (float): Margin below which POSITIVE and NEGATIVE are considered nearly equal.

    Returns:
    - overall_sentiment (str): Final sentiment (POSITIVE, NEGATIVE, or NEUTRAL).
    - sentiment_scores (dict): Aggregated scores for each sentiment.
    """
    sentiment_scores = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}

    # Aggregate scores
    for result in results:
        label = result["label"].upper()  # Ensure label is in uppercase
        if label in sentiment_scores:
            sentiment_scores[label] += result["score"]

    # Determine overall sentiment
    pos_score = sentiment_scores["POSITIVE"]
    neg_score = sentiment_scores["NEGATIVE"]
    neu_score = sentiment_scores["NEUTRAL"]

    # Logic to handle near-equal POSITIVE and NEGATIVE scores
    if abs(pos_score - neg_score) >= threshold:
        # Only return NEUTRAL if it is significantly higher
        if neu_score > 10 * pos_score and neu_score > 10 * neg_score:
            overall_sentiment = "NEUTRAL"
        else:
            overall_sentiment = "POSITIVE" if pos_score > neg_score else "NEGATIVE"
    else:
        # Return the dominating sentiment
        overall_sentiment = "NEUTRAL"

    return overall_sentiment, sentiment_scores



# Process each company
for company, ticker in tnc.items():
    input_file = f"Summary/{company}_summaries.txt"
    output_file = f"Results/{ticker}/res_sentiment.txt"

    # Read the input file
    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found. Skipping...")
        continue

    with open(input_file, "r") as file:
        text = file.read()

    # Split text into chunks
    chunks = split_text_into_chunks(text)

    # Analyze sentiment for each chunk
    chunk_results = []
    for chunk in chunks:
        result = finbert_pipeline(chunk)
        chunk_results.extend(result)

    # Aggregate sentiment scores
    overall_sentiment, sentiment_scores = aggregate_sentiments(chunk_results)

    # Save results to the output file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as file:
        file.write("Sentiment Analysis Results:\n")
        for chunk, result in zip(chunks, chunk_results):
            file.write(f"Chunk: {chunk}\n")
            file.write(f"Sentiment: {result['label']}, Score: {result['score']:.2f}\n\n")
        
        file.write("Overall Sentiment:\n")
        file.write(f"Sentiment: {overall_sentiment}\n")
        file.write(f"Scores: {sentiment_scores}\n")

    print(f"Sentiment analysis for {company} saved to {output_file}")
