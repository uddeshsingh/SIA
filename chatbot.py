import os
import logging
import discord
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from rapidfuzz import fuzz, process

# Retrieve environment variables
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DiscordBot")

#company dictionary

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

id_company = "unknown"

def get_best_match(company_name, choices):
    # Perform fuzzy matching
    result = process.extractOne(company_name.lower(), choices, scorer=fuzz.partial_token_sort_ratio)
    best_match, score = result[0], result[1]
    print( best_match, company_name, score)
    return best_match if score > 90 else "unknown"

# Initialize the Groq chat model
chat_model = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0.5,
    groq_api_key=GROQ_API_KEY
)

# Intent recognition prompt
intent_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant that classifies user intents based on the provided message. Your response must be exactly one word from the following list of intents:\n"
               "'financial_summary', 'risk_factors', 'management_analysis', 'revenue_analysis', 'expense_analysis', "
               "'liquidity_analysis', 'future_outlook', 'competitive_position', 'debt_analysis', 'segment_analysis'.\n"
               "If the message does not fit any of these intents, output 'unknown' only."),
    ("human", "{input}")
])

conversation_history_dir = "user_histories"
os.makedirs(conversation_history_dir, exist_ok=True)

# Intent-specific prompts
intent_prompts = {
    "financial_summary": ChatPromptTemplate.from_messages([
        ("system", "You are an expert in analyzing financial reports. Summarize the key financial highlights. Use the provided context to generate your response: {context}."),
        ("human", "{input}")
    ]),
    "risk_factors": ChatPromptTemplate.from_messages([
        ("system", "You are an expert in financial risk analysis. Identify and summarize the main risk factors. Use the provided context to generate your response: {context}."),
        ("human", "{input}")
    ]),
    "management_analysis": ChatPromptTemplate.from_messages([
        ("system", "You are an expert in financial management analysis. Provide insights based on the Management Discussion and Analysis section. Use the provided context to generate your response: {context}."),
        ("human", "{input}")
    ]),
    "revenue_analysis": ChatPromptTemplate.from_messages([
        ("system", "You are an expert in revenue analysis. Analyze the revenue details and explain any trends or insights. Use the provided context to generate your response: {context}."),
        ("human", "{input}")
    ]),
    "expense_analysis": ChatPromptTemplate.from_messages([
        ("system", "You are an expert in financial analysis. Break down and analyze the expense-related information. Use the provided context to generate your response: {context}."),
        ("human", "{input}")
    ]),
    "liquidity_analysis": ChatPromptTemplate.from_messages([
        ("system", "You are an expert in liquidity and solvency analysis. Review the liquidity-related details and explain the company's financial health. Use the provided context to generate your response: {context}."),
        ("human", "{input}")
    ]),
    "future_outlook": ChatPromptTemplate.from_messages([
        ("system", "You are an expert in forecasting and future financial planning. Summarize the company's outlook based on the Forward-Looking Statements section. Use the provided context to generate your response: {context}."),
        ("human", "{input}")
    ]),
    "competitive_position": ChatPromptTemplate.from_messages([
        ("system", "You are an expert in competitive analysis. Analyze the company's competitive position and market strategy. Use the provided context to generate your response: {context}."),
        ("human", "{input}")
    ]),
    "debt_analysis": ChatPromptTemplate.from_messages([
        ("system", "You are a financial analyst specializing in debt. Review the company's debt-related details and provide insights on its leverage and repayment capabilities. Use the provided context to generate your response: {context}."),
        ("human", "{input}")
    ]),
    "segment_analysis": ChatPromptTemplate.from_messages([
        ("system", "You are an expert in segment analysis. Analyze the performance of different business segments. Use the provided context to generate your response: {context}."),
        ("human", "{input}")
    ]),
    "unknown": ChatPromptTemplate.from_messages([
        ("system", "You are an expert in financial analysis. Respond to the input. If the input is within your expertise, continue the conversation using the provided context: {context}. If not, bring the topic back to your main objective. Keep your answers very concise."),
        ("human", "{input}")
    ])
}



# Initialize Discord bot
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
bot = discord.Client(intents=intents)


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore_path = "faiss_index"


def load_history(user_id):
    """Load user history from a file."""
    history_file = os.path.join(conversation_history_dir, f"{user_id}.txt")
    if os.path.exists(history_file):
        with open(history_file, "r") as file:
            return file.readlines()
    return []

def save_history(user_id, history):
    """Save user history to a file."""
    history_file = os.path.join(conversation_history_dir, f"{user_id}.txt")
    with open(history_file, "w") as file:
        file.writelines(history)


def identify_company(user_id, input_text):
    """Identify company name using fuzzy matching and LLM based on conversation history."""
    # Predefined list of company names (choices)
    choices = list(tnc.keys())

    # Attempt fuzzy matching first
    best_match = get_best_match(input_text, choices)
    if best_match != "unknown":
        logger.info(f"Company identified using fuzzy matching: {best_match}")
        return best_match

    # If fuzzy matching fails, fall back to LLM-based identification
    history = load_history(user_id)
    full_context = "\n".join(history + [f"User: {input_text}"])
    company_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant that identifies the company being discussed based on the user's input and previous conversation. Output only the company name or 'unknown'."),
        ("human", full_context)
    ])
    chain = company_prompt | chat_model
    try:
        response = chain.invoke({"input": full_context})
        llm_output = response.content.strip()
        logger.info(f"Company identified using LLM: {llm_output}")

        # Run LLM output through fuzzy matching to refine the result
        refined_match = get_best_match(llm_output, choices)
        if refined_match != "unknown":
            logger.info(f"LLM output refined using fuzzy matching: {refined_match}")
            return refined_match

        # If no valid match found, return the LLM output or unknown
        return llm_output if llm_output in choices else "unknown"
    except Exception as e:
        logger.error(f"Error identifying company: {e}")
        return "unknown"


def create_faiss_for_company(company_name):
    """Load or create a FAISS index for the specified company."""
    company_folder = os.path.join("Results", company_name)
    if not os.path.exists(company_folder):
        return None

    vectorstore_path = os.path.join("faiss_indexes", company_name)
    os.makedirs("faiss_indexes", exist_ok=True)
    
    faiss_index_file = os.path.join(vectorstore_path, "index.faiss")
    if os.path.exists(faiss_index_file):
        return FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
    else:
        loader = DirectoryLoader(company_folder, glob="**/*.txt")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(texts, embeddings)
        vectorstore.save_local(vectorstore_path)
        return vectorstore

def retrieve_relevant_docs(vectorstore, query):
    """Retrieve relevant documents using the FAISS index."""
    if not vectorstore:
        return "No relevant documents found."
    docs = vectorstore.similarity_search(query, k=3)

    for doc in docs:
        print(f"Retrieved document: {doc.metadata.get('source', 'Unknown Source')}")


    return "\n\n".join([doc.page_content for doc in docs])

# Helper function for invoking a chain
def invoke_chain_with_history(prompt_template, user_id, input_text, context= ""):
    # Retrieve or initialize user history
    history = load_history(user_id)
    
    # Combine conversation history with the current input
    full_context = "\n".join(history + [f"User: {input_text}"])
    try:
        chain = prompt_template | chat_model
        response = chain.invoke({
            "input": input_text,  # User's query
            "context": context    # Retrieved context
        })
        
        # Update conversation history
        history.append(f"User: {input_text}")
        history.append(f"Bot: {response.content.strip()}")
                
        return response.content.strip()
    except Exception as e:
        logger.error(f"Error invoking chain with history: {e}")
        return "Sorry, I couldn't process your request right now."


user_companies = {}

# Event: Bot is ready
@bot.event
async def on_ready():
    logger.info(f"Bot is online as {bot.user}")
    print(f"Bot is online as {bot.user}")

# Event: Message received
@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    # Check if the bot is mentioned
    if bot.user in message.mentions:
        user_input = message.content.replace(f"<@{bot.user.id}>", "").strip()

        user_id = str(message.author.id)
        
        # Handle reset command
        if user_input.lower() == "!reset":
            # Reset conversation history and company
            history_file = os.path.join(conversation_history_dir, f"{user_id}.txt")
            if os.path.exists(history_file):
                os.remove(history_file)
            user_companies.pop(user_id, None)
            await message.channel.send("Your conversation history and company have been reset.")
            return

        # Load user history
        history = load_history(user_id)
        
        # Recognize intent
        intent = invoke_chain_with_history(intent_prompt, message.author.id, user_input)
        logger.info(f"The intent is: {intent}")


        for i in tnc.keys():
            if i in user_input.lower():
                identified_company = identify_company(user_id, user_input)
                logger.info(f"Identified company: {identified_company}")
                if identified_company != "unknown":
                    user_companies[user_id] = identified_company
                break

        # Identify company name
        current_company = user_companies.get(user_id, "unknown")
        if current_company == "unknown":
            await message.channel.send("No company is currently set. Please set a company first by saying something like 'Change company to Tesla'.")
            return

        id_company_ticker = tnc.get(current_company, "unknown")
        vectorstore = create_faiss_for_company(id_company_ticker)

        if not vectorstore:
            await message.channel.send(f"No relevant documents found for {current_company}.")
            return

        # Retrieve relevant context
        context = retrieve_relevant_docs(vectorstore, user_input)

        print("\n\n\n\n"+context)

        # Respond based on intent
        if intent in intent_prompts:
            response = invoke_chain_with_history(intent_prompts[intent], user_id, user_input)
        else:
            response = invoke_chain_with_history(intent_prompts["unknown"], user_id, user_input)

        # Update conversation history
        history = load_history(user_id)
        history.append(f"User: {user_input}\n")
        history.append(f"Bot: {response}\n")
        save_history(user_id, history)

        await message.channel.send(response)

# Run the bot
try:
    bot.run(DISCORD_TOKEN)
except Exception as e:
    logger.error(f"Failed to start bot: {e}")