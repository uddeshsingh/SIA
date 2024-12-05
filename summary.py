import re
import os
from PyPDF2 import PdfReader
from langchain_groq import ChatGroq
from langchain_google_vertexai import VertexAI

from langchain_core.prompts import ChatPromptTemplate

# # Initialize Groq API
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# chat_model = ChatGroq(
#     model="mixtral-8x7b-32768",
#     temperature=0.0,
#     groq_api_key=GROQ_API_KEY
# )

chat_model = VertexAI(
    model="gemini-1.0-pro-002",
    temperature=0.3,
    max_output_tokens=512,
    top_p=0.3
)
# Step 1: Extract Company Name from the First Page
def extract_company_name(file_path):
    """
    Extract the company name from the first page of the 10-K report.
    """
    reader = PdfReader(file_path)
    first_page_text = reader.pages[0].extract_text()
    # print(first_page_text)
    match = re.search(r"(?i)\b(company|corporation|Corporaon|inc|ltd|limited|plc|corp)\b", first_page_text)
    if match:
        print("Yes Match")
        lines = first_page_text.splitlines()
        for line in lines:
            if match.group() in line:
                return line.split()[0].strip(' ,!')
    return "Oracle"

# Step 2: Extract Table of Contents Starting from Page 3
def extract_toc_page(file_path):
    reader = PdfReader(file_path)
    toc_text = ""
    for i in range(1, 4):  # Adjust range as needed
        toc_text += reader.pages[i].extract_text()
    return toc_text

# Step 3: Extract Headings and Page Numbers Using LLM
def extract_headings_and_pages_from_toc(toc_text):
    """
    Extract headings and page numbers from ToC text using regex.
    """
    section_page_map = {}
    # Regex pattern for headings and page numbers
    # [0-9]*\.? PART [IVXLCDM]+, Item [0-9]+[A-Z]?\. (.*?) - Page (\d+) |
    pattern = r"\*? (.*?) - (\d+)"
    matches = re.findall(pattern, toc_text)

    for match in matches:
        print(match)
        heading, page = match
        section_page_map[heading.strip()] = int(page)  # Store in dictionary
    return section_page_map

def get_top_important_headings(headings_dict):
    # Convert the headings dictionary to a string for LLM input
    headings_text = "\n".join([f"Heading: {heading.strip('*')} - Page {page}" for heading, page in headings_dict.items()])

    print(headings_text)

    # LLM prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a financial analyst trained to prioritize sections of a 10-K report. Based on their importance to understanding a company's financial health and operations, identify the top 7 most critical sections in format: Heading - Page. No need for any additional information or explanation. stick to the instructions."),
        ("human", headings_text)
    ])

    chain = prompt | chat_model

    try:
        # Get the LLM response
        response = chain.invoke({"input": headings_text})
        response_content = response

        # Debug: Print raw LLM output
        print("\nLLM Raw Output:")
        print(response_content)

        cleaned_text = re.sub(r"\*\*", "", response_content)

        print("\nLLM Processed Output:")
        print(cleaned_text)

        # Parse the response into a dictionary
        top_headings = {}
        matches = re.findall(r"^\d+\.\s+(.+?)\s+-\s+Page\s+(\d+)", cleaned_text, re.MULTILINE)
        for match in matches:
            print(match)
            heading, page = match
            top_headings[heading] = int(page)

        return top_headings
    except Exception as e:
        print(f"Error identifying top headings: {e}")
        return {}

def extract_headings_and_pages(toc_text):
    """
    Extract headings and their page numbers from LLM-processed ToC text.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant trained to identify section headings and their corresponding page numbers in financial reports. Extract headings with their starting page numbers from the following Table of Contents in format: [Heading] - [Page]."),
        ("human", f"{toc_text}")
    ])
    chain = prompt | chat_model
    try:
        # Get LLM response
        response = chain.invoke({"input": toc_text})

        # Print LLM raw output for debugging
        print("\nLLM Raw Output:")
        print(response)

        # Use regex to extract headings and page numbers
        section_page_map = extract_headings_and_pages_from_toc(response)

        # Debug: Print the dictionary
        print("\nExtracted Dictionary of Headings and Pages:")
        for heading, page in section_page_map.items():
            print(f"Heading: {heading}, Page: {page}")

        return get_top_important_headings(section_page_map)
    except Exception as e:
        print(f"Error extracting headings and pages: {e}")
        return {}

# Step 4: Preprocess Raw Text
def preprocess_text(raw_text):
    """
    Preprocess raw text by removing unnecessary whitespace, fixing encoding issues,
    and standardizing formatting.
    """
    # Remove extra whitespace
    processed_text = re.sub(r"\s+", " ", raw_text)

    # Remove non-printable characters
    processed_text = re.sub(r"[^\x20-\x7E]", "", processed_text)

    # Standardize formatting (e.g., fixing bullet points, line breaks)
    processed_text = processed_text.replace("\n", " ").strip()

    return processed_text

# Step 5: Divide Text into Chunks by Page Numbers
def chunk_10k_report_by_pages(file_path, section_page_map):
    """
    Divide the 10-K report into chunks based on page numbers from the Table of Contents.
    Split chunks further if they exceed the LLM token limit.
    """
    reader = PdfReader(file_path)
    total_pages = len(reader.pages)
    chunks = {}
    sorted_sections = sorted(section_page_map.items(), key=lambda x: x[1])  # Sort by page number

    print("\nExtracted Headings and Page Numbers:")
    for section, page in sorted_sections:
        print(f"Heading: {section}, Starting Page: {page}")

    token_limit = 4000  # Set token limit to half

    for i, (section, start_page) in enumerate(sorted_sections):
        end_page = (
            sorted_sections[i + 1][1] - 1
            if i + 1 < len(sorted_sections)
            else total_pages
        )
        chunk_text = ""
        for page_num in range(start_page - 1, end_page):  # Adjust for 0-index
            chunk_text += reader.pages[page_num].extract_text()
        
        # Preprocess the chunk text
        chunk_text = preprocess_text(chunk_text)

        # Split into smaller chunks if exceeding token limit
        words = chunk_text.split()
        if len(words) > token_limit:
            for j in range(0, len(words), token_limit):
                sub_chunk = " ".join(words[j:j + token_limit])
                chunks[f"{section} (Part {j // token_limit + 1})"] = sub_chunk
        else:
            chunks[section] = chunk_text

    # Debug: Print the start of each chunk
    print("\nPreview of chunks by page numbers:")
    for section, content in chunks.items():
        print(f"Section: {section}")
        print(f"Start of Chunk: {content[:200]}...\n")  # Print the first 200 characters of each chunk

    return chunks

# Step 6: Summarize Content of Each Chunk Using LLM
def summarize_chunk_with_groq(chunk_name, text_chunk, company_name):
    """
    Pass a chunk to Groq and generate a summary. Include important financial metrics and company name.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are a financial analyst specializing in corporate filings. The company being analyzed is {company_name}. Summarize the following content concisely, highlighting key financial metrics such as EBITDA, Assets, Liabilities, Growth, and any other critical figures."),
        ("human", f"{text_chunk}")
    ])
    chain = prompt | chat_model
    try:
        response = chain.invoke({"input": text_chunk})
        response_content = response
        print(f"\nSummary for {chunk_name}:")
        print(response_content)
        return response_content
    except Exception as e:
        print(f"Error summarizing chunk {chunk_name}: {e}")
        return ""

# Step 7: Main Workflow
def process_10k_report(file_path, company_name):
    """
    Process a 10-K report to summarize content by headings.
    """
    print(f"Company Name: {company_name}")

    print("Extracting Table of Contents from the 10-K report starting from page 3...")
    toc_text = extract_toc_page(file_path)

    print("Extracting headings and page numbers from the Table of Contents...")
    section_page_map = extract_headings_and_pages(toc_text)

    if not section_page_map:
        print("No headings or page numbers could be extracted. Exiting.")
        return {}

    print("\nChunking the 10-K report based on page numbers...")
    chunks = chunk_10k_report_by_pages(file_path, section_page_map)

    summaries = {}

    for heading, chunk in chunks.items():
        print(f"Processing chunk: {heading}")
        summary = summarize_chunk_with_groq(heading, chunk, company_name)
        summaries[heading] = summary

    print("Summarization complete.")
    return summaries

# File path to the 10-K report
file_path = "Reports/10-K/msft-10k-2024.pdf"

print("Extracting company name from the 10-K report...")
company_name = extract_company_name(file_path)

# Run the workflow
summaries = process_10k_report(file_path, company_name)

# Step 8: Save Summaries to a Text File
def save_summaries_to_file(summaries, company_name):
    """
    Save the summaries into a text file named after the company.
    """
    cn = company_name.replace(' ', '_').lower()
    filename = f"Summary/{cn}_summaries.txt"
    try:
        with open(filename, "w", encoding="utf-8") as file:
            for heading, summary in summaries.items():
                file.write(f"Heading: {heading}\n")
                file.write(f"Summary: {summary}\n")
                file.write("\n" + "="*80 + "\n\n")  # Separator for readability
        print(f"Summaries successfully saved to {filename}")
    except Exception as e:
        print(f"Error saving summaries to file: {e}")

# Save the results
save_summaries_to_file(summaries, company_name)


# Output results
print("\nSummarized Content:")
for heading, summary in summaries.items():
    print(f"Heading: {heading}")
    print(f"Summary: {summary}\n")
