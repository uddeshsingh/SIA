import requests
import json
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Initialize Groq API
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
chat_model = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0.2,
    groq_api_key=GROQ_API_KEY
)

# Define the directory where data will be cached
CACHE_DIR = "financial_data_cache"

# Step 1: Check if the data is already cached, else fetch from API
def fetch_financial_data(ticker):
    """
    Fetch financial data from local cache or API using the ticker as the file name.
    """
    file_name = f"{ticker}_financial_data.json"
    cache_path = os.path.join(CACHE_DIR, file_name)
    
    if os.path.exists(cache_path):
        print(f"Loading cached data for {ticker}...")
        with open(cache_path, 'r') as file:
            data = json.load(file)
    else:
        print(f"Fetching financial data for {ticker} from API...")
        api_url = f"https://api.datajockey.io/v0/company/financials?apikey=1a31c519e4abb707d72ab5770580c8624b925ad99c47d2e7817f&ticker={ticker}"
        response = requests.get(api_url)
        data = response.json()  # Parse the JSON response
        
        # Save the fetched data to a file named by the ticker
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(cache_path, 'w') as file:
            json.dump(data, file, indent=4)
    return data

# Step 2: Process the financial data dynamically for the last 7 years
def process_financial_data(data):
    """
    Process the financial data to extract only the last 7 years of data dynamically.
    """
    company_info = data['company_info']
    financial_data = data['financial_data']['annual']

    # Filter for the last 7 years of data
    last_7_years = ['2017', '2018', '2019', '2020', '2021', '2022', '2023']
    
    # Dynamically extract all the metrics (like revenue, cost_of_revenue, net_income, etc.)
    metrics = {}
    
    # Loop through each key in the 'financial_data' section
    for category, values in financial_data.items():
        # Only keep data for the last 7 years
        metrics[category] = {year: values[year] for year in last_7_years if year in values}

    return company_info, metrics


def convert_to_groq_input(data):
    result = []

    for metric, values in data.items():
        metric_string = f"**{metric.replace('_', ' ').capitalize()}:**"
        for year, value in values.items():
            metric_string += f"\n  - Year {year}: ${value:,}"
        result.append(metric_string)
    
    return result

# Step 3: Split structured data into chunks based on financial headings
def chunk_data_by_heading(metrics):
    structured_data_list = convert_to_groq_input(metrics)

    return structured_data_list

# Step 4: Create Groq Prompt for Inference
def create_groq_prompt(structured_data):
    """
    Create a prompt for Groq to analyze the dynamic financial data.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant trained to analyze financial data. Based on the structured data provided, give me detailed inference pertaining to investing over the last 7 years concisely."),
        ("human", f"{structured_data}")
    ])
    return prompt

# Step 5: Process the Structured Data with Groq
def process_with_groq(structured_data):
    """
    Pass the structured data to Groq for inference and analysis.
    """
    prompt = create_groq_prompt(structured_data)
    chain = prompt | chat_model
    try:
        response = chain.invoke({"input": structured_data})
        return response.content  # Groq's output will contain analysis of the data
    except Exception as e:
        print(f"Error during Groq processing: {e}")
        return "Error during analysis."

# Step 6: Main Workflow for Fetching, Processing, and Analyzing Data
def analyze_financial_performance(ticker):
    """
    Fetch the financial data from the API (or local cache), process it for the last 7 years, and analyze performance using Groq.
    """
    # Step 1: Fetch financial data for the given ticker (e.g., TSLA)
    print(f"Fetching financial data for {ticker}...")
    data = fetch_financial_data(ticker)

    # Step 2: Process the data dynamically for the last 7 years
    print("Processing financial data...")
    company_info, metrics = process_financial_data(data)

    # Step 3: Chunk the structured data by heading (e.g., revenue, gross profit)
    print("Chunking structured data by heading...")
    data_chunks = chunk_data_by_heading(metrics)

    # Step 4: Analyze each chunk of financial data using Groq
    print("Analyzing data using Groq...")
    analysis_results = []
    for chunk in data_chunks:
        print(chunk)
        analysis_result = process_with_groq(chunk)
        analysis_results.append(analysis_result)

    # Step 5: Output the analysis results and save to file
    results_dir = f"Results/{ticker}"
    os.makedirs(results_dir, exist_ok=True)

    analysis_file_path = os.path.join(results_dir, "analysis_results.txt")
    with open(analysis_file_path, 'w') as result_file:
        result_file.write("\n\n".join(analysis_results))

    print(f"Analysis results saved to {analysis_file_path}")

# Run the workflow for a company (for example, TSLA)
ticker = "tsla"  # Replace with the ticker of your choice
analyze_financial_performance(ticker)
