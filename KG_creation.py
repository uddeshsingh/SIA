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

# Step 1: Extract Table of Contents Starting from Page 3

def extract_toc_page(file_path):
    reader = PdfReader(file_path)
    toc_text = ""
    for i in range(2, 5):  # Adjust range as needed
        toc_text += reader.pages[i].extract_text()
    return toc_text

# Step 2: Extract Headings and Page Numbers Using LLM


def extract_headings_and_pages_from_toc(toc_text):
    """
    Extract headings and page numbers from ToC text using regex.
    """
    section_page_map = {}
    # Regex pattern for headings and page numbers
    pattern = r"PART [IVXLCDM]+, Item [0-9]+[A-Z]?\. (.*?) - Page (\d+)"
    matches = re.findall(pattern, toc_text)

    for match in matches:
        heading, page = match
        section_page_map[heading.strip()] = int(page)  # Store in dictionary
    return section_page_map

def extract_headings_and_pages(toc_text):
    """
    Extract headings and their page numbers from LLM-processed ToC text.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant trained to identify section headings and their corresponding page numbers in financial reports. Extract headings with their starting page numbers from the following Table of Contents."),
        ("human", f"{toc_text}")
    ])
    chain = prompt | chat_model
    try:
        # Get LLM response
        response = chain.invoke({"input": toc_text})

        # Print LLM raw output for debugging
        print("\nLLM Raw Output:")
        print(response.content)

        # Use regex to extract headings and page numbers
        section_page_map = extract_headings_and_pages_from_toc(response.content)

        # Debug: Print the dictionary
        print("\nExtracted Dictionary of Headings and Pages:")
        for heading, page in section_page_map.items():
            print(f"Heading: {heading}, Page: {page}")

        return section_page_map
    except Exception as e:
        print(f"Error extracting headings and pages: {e}")
        return {}


# Step 3: Divide Text into Chunks by Page Numbers

def chunk_10k_report_by_pages(file_path, section_page_map):
    """
    Divide the 10-K report into chunks based on page numbers from the Table of Contents.
    """
    reader = PdfReader(file_path)
    total_pages = len(reader.pages)
    chunks = {}
    sorted_sections = sorted(section_page_map.items(), key=lambda x: x[1])  # Sort by page number

    print("\nExtracted Headings and Page Numbers:")
    for section, page in sorted_sections:
        print(f"Heading: {section}, Starting Page: {page}")

    for i, (section, start_page) in enumerate(sorted_sections):
        end_page = (
            sorted_sections[i + 1][1] - 1
            if i + 1 < len(sorted_sections)
            else total_pages
        )
        chunk_text = ""
        for page_num in range(start_page - 1, end_page):  # Adjust for 0-index
            chunk_text += reader.pages[page_num].extract_text()
        chunks[section] = chunk_text

    # Debug: Print the start of each chunk
    print("\nPreview of chunks by page numbers:")
    for section, content in chunks.items():
        print(f"Section: {section} (Pages {section_page_map[section]}-{end_page})")
        print(f"Start of Chunk: {content[:200]}...\n")  # Print the first 200 characters of each chunk

    return chunks

# Step 4: Define Groq Prompt for NER and Relationship Extraction

def create_prompt(text_chunk):
    """
    Create a Groq prompt for extracting entities and relationships from a chunk of text.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert in financial analysis and knowledge graph generation. Extract entities (nodes) and relationships (edges) from the input text, this way, node1 -> relationship -> node2"),
        ("human", f"{text_chunk}")
    ])
    return prompt

# Step 5: Process Chunks Using Groq

def process_chunk_with_groq(chunk_name, text_chunk):
    """
    Pass a chunk to Groq and parse its output into nodes and edges.
    """
    prompt = create_prompt(text_chunk)
    chain = prompt | chat_model
    try:
        response = chain.invoke({"input": text_chunk})
        response_content = response.content
        print("response content: " , response_content)

        # Parse nodes and edges from Groq's response (example assumes JSON-like output)
        nodes = []
        edges = []
        for line in response_content.split("\n"):
            if line.startswith("Node:"):
                nodes.append(line.replace("Node:", "").strip())
            elif line.startswith("Edge:"):
                edges.append(line.replace("Edge:", "").strip())

        return nodes, edges
    except Exception as e:
        print(f"Error processing chunk {chunk_name}: {e}")
        return [], []

# Step 6: Main Workflow

def process_10k_report(file_path):
    """
    Process a 10-K report to extract entities and relationships.
    """
    print("Extracting Table of Contents from the 10-K report starting from page 3...")
    toc_text = extract_toc_page(file_path)

    print("Extracting headings and page numbers from the Table of Contents...")
    section_page_map = extract_headings_and_pages(toc_text)

    if not section_page_map:
        print("No headings or page numbers could be extracted. Exiting.")
        return [], []

    print("\nChunking the 10-K report based on page numbers...")
    chunks = chunk_10k_report_by_pages(file_path, section_page_map)

    all_nodes = []
    all_edges = []

    for heading, chunk in chunks.items():
        print(f"Processing chunk: {heading}")
        nodes, edges = process_chunk_with_groq(heading, chunk)
        all_nodes.extend(nodes)
        all_edges.extend(edges)

    print("Extraction complete.")
    return all_nodes, all_edges

# File path to the 10-K report
file_path = "Reports/10-K/tsla-10k-2024.pdf"

# Run the workflow
nodes, edges = process_10k_report(file_path)

# Output results
print("Extracted Nodes:", nodes)
print("Extracted Edges:", edges)
