import re
import os
from PyPDF2 import PdfReader
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Initialize Groq API
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
chat_model = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0.4,
    groq_api_key=GROQ_API_KEY
)

# Function to get top 5 important headings
def get_top_important_headings(headings_dict):
    """
    Uses the LLM to identify the top 5 important headings from a dictionary of headings and page numbers.
    
    Args:
        headings_dict (dict): A dictionary where keys are headings and values are page numbers.

    Returns:
        dict: A dictionary of the top 5 important headings with their page numbers.
    """
    # Convert the headings dictionary to a string for LLM input
    headings_text = "\n".join([f"Heading: {heading}, Starting Page: {page}" for heading, page in headings_dict.items()])

    # LLM prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a financial analyst trained to prioritize sections of a 10-K report. Based on their importance to understanding a company's financial health and operations, identify the top 5 most critical sections from the following list. Return only the headings and their page numbers in the original format."),
        ("human", headings_text)
    ])

    chain = prompt | chat_model

    try:
        # Get the LLM response
        response = chain.invoke({"input": headings_text})
        response_content = response.content

        # Debug: Print raw LLM output
        print("\nLLM Raw Output:")
        print(response_content)

        # Parse the response into a dictionary
        top_headings = {}
        matches = re.findall(r"Heading: (.*?), Starting Page: (\d+)", response_content)
        for match in matches:
            heading, page = match
            top_headings[heading] = int(page)

        return top_headings
    except Exception as e:
        print(f"Error identifying top headings: {e}")
        return {}

# Example usage
headings_dict = {
    "Business": 4,
    "Risk Factors": 14,
    "Unresolved Staff Comments": 28,
    "Cybersecurity": 29,
    "Properties": 30,
    "Legal Proceedings": 30,
    "Mine Safety Disclosures": 30,
    "Market for Registrant's Common Equity, Related Stockholder Matters and Issuer Purchases of Equity Securities": 31,
    "Management's Discussion and Analysis of Financial Condition and Results of Operations": 33,
    "Quantitative and Qualitative Disclosures about Market Risk": 45,
    "Financial Statements and Supplementary Data": 46,
    "Changes in and Disagreements with Accountants on Accounting and Financial Disclosure": 93,
    "Controls and Procedures": 93,
    "Other Information": 94,
    "Disclosure Regarding Foreign Jurisdictions that Prevent Inspections": 94,
    "Directors, Executive Officers and Corporate Governance": 95,
    "Executive Compensation": 95,
    "Security Ownership of Certain Beneficial Owners and Management and Related Stockholder Matters": 95,
    "Certain Relationships and Related Transactions, and Director Independence": 95,
    "Principal Accountant Fees and Services": 95,
    "Exhibits and Financial Statement Schedules": 96,
    "Summary": 111
}

# Get the top 5 headings
top_important_headings = get_top_important_headings(headings_dict)

# Output results
print("\nTop 5 Important Headings:")
for heading, page in top_important_headings.items():
    print(f"Heading: {heading}, Starting Page: {page}")
