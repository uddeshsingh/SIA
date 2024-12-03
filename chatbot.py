import os
import logging
import discord
from langchain_google_vertexai import VertexAI
# from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from rapidfuzz import fuzz, process

# Retrieve environment variables
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
# VERTEX_AI_API_KEY = os.getenv("VERTEX_AI_API_KEY")

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
    return best_match if score > 90 else "unknown"


chat_model = VertexAI(
    model="gemini-1.0-pro-002",
    temperature=0.3,
    max_output_tokens=512,
    top_p=0.9
)
# chat_model1 = ChatOpenAI(
#     model="gpt-3.5-turbo",
#     temperature=0.5,
#     max_completion_tokens=512,
#     top_p=0.9

# )

# Intent recognition prompt
intent_prompt = ChatPromptTemplate.from_messages([
    ("system", "Your name is SIA (Stock Investment Advisor). You are not allowed to mention Gemini or any other identity. You must always respond as SIA, an assistant that classifies user intents based on the provided message. Your response must be exactly one word from the following list of intents:\n"
               "'financial_summary', 'risk_factors', 'management_analysis', 'revenue_analysis', 'expense_analysis', "
               "'liquidity_analysis', 'future_outlook', 'competitive_position', 'debt_analysis', 'segment_analysis',"
               "'news_summary', 'financial_event'.\n"
               "If the message does not fit any of these intents, output 'unknown' only."),
    ("human", "{input}")
])

conversation_history_dir = "user_histories"
os.makedirs(conversation_history_dir, exist_ok=True)

# Intent-specific prompts
intent_prompts = {
    "financial_summary": ChatPromptTemplate.from_messages([
        ("system", "Your name is SIA (Stock Investment Advisor). You are not allowed to mention Gemini or any other identity. You must always respond as SIA, an expert in analyzing financial reports. Summarize the key financial highlights. CONTEXT START: {context} CONTEXT END. Keep your answers within 1500 characters"),
        ("human", "{input}")
    ]),
    "risk_factors": ChatPromptTemplate.from_messages([
        ("system", "Your name is SIA (Stock Investment Advisor). You are not allowed to mention Gemini or any other identity. You must always respond as SIA, an expert in financial risk analysis. Identify and summarize the main risk factors. CONTEXT START: {context} CONTEXT END. Keep your answers within 1500 characters"),
        ("human", "{input}")
    ]),
    "management_analysis": ChatPromptTemplate.from_messages([
        ("system", "Your name is SIA (Stock Investment Advisor). You are not allowed to mention Gemini or any other identity. You must always respond as SIA, an expert in financial management analysis. Provide insights based on the Management Discussion and Analysis section. CONTEXT START: {context} CONTEXT END. Keep your answers within 1500 characters"),
        ("human", "{input}")
    ]),
    "revenue_analysis": ChatPromptTemplate.from_messages([
        ("system", "Your name is SIA (Stock Investment Advisor). You are not allowed to mention Gemini or any other identity. You must always respond as SIA, an expert in revenue analysis. Analyze the revenue details and explain any trends or insights. CONTEXT START: {context} CONTEXT END. Keep your answers within 1500 characters"),
        ("human", "{input}")
    ]),
    "expense_analysis": ChatPromptTemplate.from_messages([
        ("system", "Your name is SIA (Stock Investment Advisor). You are not allowed to mention Gemini or any other identity. You must always respond as SIA, an expert in financial analysis. Break down and analyze the expense-related information. CONTEXT START: {context} CONTEXT END. Keep your answers within 1500 characters"),
        ("human", "{input}")
    ]),
    "liquidity_analysis": ChatPromptTemplate.from_messages([
        ("system", "Your name is SIA (Stock Investment Advisor). You are not allowed to mention Gemini or any other identity. You must always respond as SIA, an expert in liquidity and solvency analysis. Review the liquidity-related details and explain the company's financial health. CONTEXT START: {context} CONTEXT END. Keep your answers within 1500 characters"),
        ("human", "{input}")
    ]),
    "future_outlook": ChatPromptTemplate.from_messages([
        ("system", "Your name is SIA (Stock Investment Advisor). You are not allowed to mention Gemini or any other identity. You must always respond as SIA, an expert in forecasting and future financial planning. Summarize the company's outlook based on the Forward-Looking Statements section. CONTEXT START: {context} CONTEXT END. Keep your answers within 1500 characters"),
        ("human", "{input}")
    ]),
    "competitive_position": ChatPromptTemplate.from_messages([
        ("system", "Your name is SIA (Stock Investment Advisor). You are not allowed to mention Gemini or any other identity. You must always respond as SIA, an expert in competitive analysis. Analyze the company's competitive position and market strategy. CONTEXT START: {context} CONTEXT END. Keep your answers within 1500 characters"),
        ("human", "{input}")
    ]),
    "debt_analysis": ChatPromptTemplate.from_messages([
        ("system", "Your name is SIA (Stock Investment Advisor). You are not allowed to mention Gemini or any other identity. You must always respond as SIA, a financial analyst specializing in debt. Review the company's debt-related details and provide insights on its leverage and repayment capabilities. CONTEXT START: {context} CONTEXT END. Keep your answers within 1500 characters"),
        ("human", "{input}")
    ]),
    "segment_analysis": ChatPromptTemplate.from_messages([
        ("system", "Your name is SIA (Stock Investment Advisor). You are not allowed to mention Gemini or any other identity. You must always respond as SIA, an expert in segment analysis. Analyze the performance of different business segments. CONTEXT START: {context} CONTEXT END. Keep your answers within 1500 characters"),
        ("human", "{input}")
    ]),
    "news_summarize": ChatPromptTemplate.from_messages([
        ("system", "Your name is SIA (Stock Investment Advisor). You are not allowed to mention Gemini or any other identity. You must always respond as SIA, an expert in news summarizer. Summarize the news. CONTEXT START: {context} CONTEXT END. Bring the topic back to your main objective. Keep your answers very concise."),
        ("human", "{input}")
    ]),
    "financial_event": ChatPromptTemplate.from_messages([
        ("system", "Your name is SIA (Stock Investment Advisor). You are not allowed to mention Gemini or any other identity. You must always respond as SIA, an expert in event summarizer. Summarize the related events. CONTEXT START: {context} CONTEXT END. Bring the topic back to your main objective. Keep your answers very concise."),
        ("human", "{input}")
    ]),
    "unknown": ChatPromptTemplate.from_messages([
        ("system", "Your name is SIA (Stock Investment Advisor). You are not allowed to mention Gemini or any other identity. You must always respond as SIA, an expert in financial analysis. Respond to the input. CONTEXT START: {context} CONTEXT END. Bring the topic back to your main objective. Keep your answers very concise."),
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
        ("system", "Your name is SIA (Stock Investment Advisor). You are not allowed to mention Gemini or any other identity. You must always respond as SIA, an assistant that identifies the company being discussed based on the user's input and previous conversation. Output only the company name or 'unknown'."),
        ("human", full_context)
    ])
    chain = company_prompt | chat_model
    try:
        response = chain.invoke({"input": full_context})
        llm_output = response.strip()
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
        loader = DirectoryLoader(company_folder, glob=["**/*.txt", "**/*.json"])
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        texts = text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(texts, embeddings)
        vectorstore.save_local(vectorstore_path)
        return vectorstore

def retrieve_relevant_docs(vectorstore, query):
    """Retrieve relevant documents using the FAISS index."""
    if not vectorstore:
        return "No relevant documents found."

    # Perform FAISS similarity search
    docs = vectorstore.similarity_search(query, k=5)

    # Path of the main document
    main_doc_path = 'Results/tsla/analysis_results.txt'

    # Check if the main document is already in the results
    main_doc_included = any(main_doc_path in (doc.metadata.get('source', '') or '') for doc in docs)

    # Load the main document if not included
    # if not main_doc_included:
    #     try:
    #         with open(main_doc_path, "r") as file:
    #             main_doc_content = file.read()
    #         main_doc = Document(page_content=main_doc_content, metadata={"source": main_doc_path})
    #         docs.insert(0, main_doc)  # Add the main document to the top of the results
    #     except Exception as e:
    #         print(f"Error loading main document: {e}")

    # Debugging: Print retrieved document sources
    for doc in docs:
        logger.info(f"Retrieved document: {doc.metadata.get('source', 'Unknown Source')}")

    return "\n\n".join([doc.page_content for doc in docs])

# Helper function for invoking a chain
def invoke_chain_with_history(prompt_template, user_id, input_text, context= ""):
    # Retrieve or initialize user history
    history = load_history(user_id)
    
    # Combine conversation history with the current input
    full_context = "\n".join([context]+ history + [f"User: {input_text}"])
    try:
        chain = prompt_template | chat_model

        # rendered_prompt = prompt_template.format_prompt(context=context, input=input_text)
        # print(f"\n\n\n\nRendered Prompt: {rendered_prompt}")
        response = chain.invoke({
            "input": input_text,  # User's query
            "context": full_context    # Retrieved context
        })
        
        # Update conversation history
        history.append(f"User: {input_text}")
        history.append(f"Bot: {response.strip()}")

        return response.strip()
    except Exception as e:
        logger.error(f"Error invoking chain with history: {e}")
        return "Sorry, I couldn't process your request right now."

def split_message(message, limit=2000):
    """Split a message into chunks that are within the character limit."""
    return [message[i:i+limit] for i in range(0, len(message), limit)]


user_companies = {}

# Event: Bot is ready
@bot.event
async def on_ready():
    logger.info(f"Bot is online as {bot.user}")
    print(f"Bot is online as {bot.user}")

# Event: Message received
@bot.event
async def on_message(message):
    logger.info(message.content)
    if message.author == bot.user:
        return

    # Check if the bot is mentioned
    logger.info(message.mentions)
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
        print("loaded till here")

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

        # Identify company namess
        current_company = user_companies.get(user_id, "unknown")
        if current_company == "unknown":
            # Provide a generic answer for basic queries
            generic_response = "I'm here to help with general questions. For advanced analysis, please set a company by saying something like 'Change company to Tesla'."

            try:
                print(intent, intent_prompts[intent])
            except Exception as e:
                print(e)

            if intent in intent_prompts:
                try:
                    response = invoke_chain_with_history(intent_prompts[intent], user_id, user_input,"Your name is SIA (Stock Investment Advisor). You are not allowed to mention Gemini or any other identity. You must always respond as SIA. You were developed to help users understand the current landscape of top tech companies" )
                    logger.info(f"BOT RESPONSE (generic): {response}")
                except Exception as e:
                    logger.error(f"Error generating generic response: {e}")
                    response = generic_response
            else:
                response = generic_response

            # Send response
            if len(response) > 2000:
                chunks = split_message(response)
                for chunk in chunks:
                    await message.channel.send(chunk)
            else:
                await message.channel.send(response)
            return

        id_company_ticker = tnc.get(current_company, "unknown")
        vectorstore = create_faiss_for_company(id_company_ticker)

        if not vectorstore:
            await message.channel.send(f"No relevant documents found for {current_company}.")
            return

        # Retrieve relevant context
        context = retrieve_relevant_docs(vectorstore, user_input)

        # Respond based on intent
        if intent in intent_prompts:
            response = invoke_chain_with_history(intent_prompts[intent], user_id, user_input, context)
        else:
            response = invoke_chain_with_history(intent_prompts["unknown"], user_id, user_input, context)

        # Update conversation history
        history = load_history(user_id)
        history.append(f"User: {user_input}\n")
        history.append(f"Bot: {response}\n")
        save_history(user_id, history)


        logger.info(f"BOT RESPONSE: {response}")
        if len(response) > 2000:
            chunks = split_message(response)
            for chunk in chunks:
                await message.channel.send(chunk)
        else:
            await message.channel.send(response)


# Run the bot
try:
    bot.run(DISCORD_TOKEN)
except Exception as e:
    logger.error(f"Failed to start bot: {e}")