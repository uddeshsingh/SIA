import os
from langchain_google_vertexai import VertexAI

# Define the ticker symbol dictionary
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

chat_model = VertexAI(
    model="gemini-1.0-pro-002",
    temperature=0.5,
    max_output_tokens=2000,
    top_p=0.5
)

def create_prompt(text):
    return f"""
You are an advanced financial NLP system. Analyze the following text to extract:
1. Entities (such as companies, locations, people, dates, and monetary values).
2. Relationships between the extracted entities.

Text: {text}

Provide output as a list in the format (Entity1 > Relationship > Entity2).

Do not output any thing else. Just plain text.
"""

# Process each company's text file
for company, ticker in tnc.items():
    input_file = f"Summary/{company}_summaries.txt"
    output_file = f"Results/{ticker}/knowledge_graph.txt"

    # Read the input file
    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found. Skipping...")
        continue

    with open(input_file, "r") as file:
        text = file.read()

    # Generate a prompt
    prompt = create_prompt(text)

    # Query the Vertex AI model
    response = chat_model.generate([prompt])
    print(response.generations[0][0].text)
    output = response.generations[0][0].text

    # Save results to the output file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as file:
        file.write(output)

    print(f"Knowledge graph for {company} saved to {output_file}")