import re
import os
from langchain_google_vertexai import VertexAI
from langchain_core.prompts import ChatPromptTemplate

# Initialize Groq API
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
chat_model = VertexAI(
    model="gemini-1.0-pro-002",
    temperature=0.3,
    max_output_tokens=2048,
    top_p=0.3
)

def read_data_from_file(file_path):
    """
    Reads data from a text file and parses it into a list of dictionaries.

    Args:
        file_path (str): Path to the input text file.

    Returns:
        list: A list of dictionaries with 'Heading' and 'Summary' keys.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Split entries based on the separator
    entries = content.split("=" * 80)
    for entry in entries:
        if entry.strip():
            # Extract heading and summary using regex
            heading_match = re.search(r"Heading: (.*?)\n", entry)
            summary_match = re.search(r"Summary:(.*)", entry, re.DOTALL)
            
            if heading_match and summary_match:
                heading = heading_match.group(1).strip()
                summary = summary_match.group(1).strip()
                data.append({"Heading": heading, "Summary": summary})

    return data

def collate_texts_by_heading(data):
    """
    Collates texts under the same heading from a dataset.

    Args:
        data (list): A list of dictionaries with 'Heading' and 'Summary' keys.

    Returns:
        dict: A dictionary where keys are headings (without parts) and values are concatenated summaries.
    """
    collated_texts = {}
    
    for entry in data:
        heading = entry["Heading"]
        summary = entry["Summary"]
        
        # Remove part identifiers (e.g., "(Part X)") to unify headings
        unified_heading = re.sub(r"\(Part \d+\)", "", heading).strip()
        
        if unified_heading in collated_texts:
            collated_texts[unified_heading] += "\n\n" + summary  # Concatenate with a separator for readability
        else:
            collated_texts[unified_heading] = summary

    return collated_texts

def refine_with_groq(collated_texts):
    """
    Use Groq to remove redundancies and build a knowledge graph.

    Args:
        collated_texts (dict): Dictionary with headings and their collated summaries.

    Returns:
        dict: A refined dictionary with redundancies removed.
        list: A list of extracted knowledge graph entities and relationships.
    """
    refined_texts = {}
    knowledge_graph = []

    for heading, summary in collated_texts.items():
        # Define a prompt to refine text and extract knowledge graph
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a financial analyst and knowledge graph expert. Refine the input to remove redundancies and return a clear summary."),
            ("human", f"Heading: {heading}\nSummary: {summary}")
        ])
        chain = prompt | chat_model

        try:
            response = chain.invoke({"input": summary})
            response_content = response
            # print(response)

            # Separate refined summary and knowledge graph

            print(response)
            refined, *graph_lines = response_content.split("\n")
            refined_texts[heading] = response.strip()

            # print(refined_texts)

            # Extract and clean nodes and relationships
            for line in graph_lines:
                if "->" in line:
                    clean_line = re.sub(r"^[0-9]+\\.|^\\-\\s*", "", line).strip()  # Remove leading numbers, bullets
                    knowledge_graph.append(clean_line)

        except Exception as e:
            print(f"Error processing heading {heading}: {e}")
            refined_texts[heading] = summary  # Fallback to original summary

    return refined_texts, clean_knowledge_graph(knowledge_graph)

def clean_knowledge_graph(knowledge_graph):
    """
    Cleans a knowledge graph text file by removing prefixes like numbers or bullet points.

    Args:
        input_file (str): Path to the input knowledge graph file.
        output_file (str): Path to save the cleaned knowledge graph file.
    """
    cleaned_lines = []

    for line in knowledge_graph:
        # Remove leading numbers, bullet points, or whitespace
        cleaned_line = re.sub(r"^\d+\.\s*|\-\s*", "", line).strip()
        if cleaned_line:  # Ensure line isn't empty after cleaning
            cleaned_lines.append(cleaned_line)

    return cleaned_lines


def save_to_file(refined_texts, knowledge_graph, output_file, graph_file):
    """
    Saves refined texts and the knowledge graph to separate files.

    Args:
        refined_texts (dict): Dictionary with headings and refined summaries.
        knowledge_graph (list): List of knowledge graph entries.
        output_file (str): Path to save the refined summaries.
        graph_file (str): Path to save the knowledge graph.
    """

    with open(output_file, 'w', encoding='utf-8') as file:
        for heading, summary in refined_texts.items():
            file.write(f"Heading: {heading}\n")
            file.write(f"Summary:\n{summary}\n")
            file.write("=" * 80 + "\n\n")

    with open(graph_file, 'w', encoding='utf-8') as file:
        for entry in knowledge_graph:
            file.write(f"{entry}\n")


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


# Example usage
company = 'oracle'
input_file = "Summary/"+company+"_summaries.txt"  # Path to the input text file
output_file = "Results/"+tnc[company]+"/refined_summaries.txt"  # Path to the refined summaries file
graph_file = "Results/"+tnc[company]+"/knowledge_graph.txt"  # Path to the knowledge graph file

# Read data from input file
data = read_data_from_file(input_file)

# Collate texts under the same heading
collated = collate_texts_by_heading(data)

# Use Groq to refine summaries and build a knowledge graph
refined, graph = refine_with_groq(collated)
# print(collated)
# print(refined)

# Save refined summaries and knowledge graph to files
save_to_file(refined, graph, output_file, graph_file)

print(f"Refined summaries saved to {output_file}")
print(f"Knowledge graph saved to {graph_file}")
