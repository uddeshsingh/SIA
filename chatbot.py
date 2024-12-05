import os
import re
import logging
import discord
import asyncio
from langchain_google_vertexai import VertexAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from rapidfuzz import fuzz, process

# Environment setup and logging
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DiscordBot")

# Constants
STATES = {
    "GREETING": "Greeting and Introduction",
    "RANDOM_CONVO": "Random Conversation",
    "COMPANY_SELECTION": "Company Selection",
    "INTENT_IDENTIFICATION": "Intent Identification",
    "DETAILED_RESPONSE": "Detailed Response",
    "QA": "Questions",
    "WRAP_UP": "Session Wrap-Up",
    "FEEDBACK": "Feedback Loop",
    "RESET_HISTORY": "Reset the conversation history",
    "INACTIVITY": "Inactivity Handling",
    "ERROR_RECOVERY": "Error Recovery",
}
COMPANIES = {
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
    "oracle": "orcl",
}

conversation_history_dir = "user_histories"
os.makedirs(conversation_history_dir, exist_ok=True)

# State and user data storage
user_states = {}
user_companies = {}
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

user_companies = {}

# LLM setup
chat_model = VertexAI(
    model="gemini-1.0-pro-002",
    temperature=0.3,
    max_output_tokens=512,
    top_p=0.9
)

# Helper functions
def get_user_state(user_id):
    """Retrieve or initialize the user's state."""
    return user_states.setdefault(user_id, { "company": None, "intent": None})

def update_user_state(user_id, key, value):
    """Update a specific key in the user's state."""
    user_states[user_id][key] = value

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

def fuzzy_match(input_text, choices, threshold=80):
    """Perform fuzzy matching on input."""
    result = process.extractOne(input_text.lower(), choices, scorer=fuzz.partial_token_sort_ratio)
    return result[0] if result and result[1] >= threshold else "unknown"

# Function to determine next state based on LLM output
def get_next_state(llm_response):
    """
    Determine the next state based on LLM's response.
    """
    state_mapping = {
        "greeting": STATES["GREETING"],
        "random_conversation": STATES["RANDOM_CONVO"],
        "company_selection": STATES["COMPANY_SELECTION"],
        "intent_identification": STATES["INTENT_IDENTIFICATION"],
        "detailed_response": STATES["DETAILED_RESPONSE"],
        "follow_up": STATES["FOLLOW_UP"],
        "summarize": STATES["SUMMARIZE"],
        "next_steps": STATES["NEXT_STEPS"],
        "wrap_up": STATES["WRAP_UP"],
        "inactivity": STATES["INACTIVITY"],
        "error_recovery": STATES["ERROR_RECOVERY"],
        "reset_history": STATES["RESET_HISTORY"],
        "feedback": STATES["FEEDBACK"],
        "questions": STATES["QA"],
        "news_summary": STATES["NEWS_SUMMARY"],
        "fin_event": STATES["FIN_EVENT"],
    }
    return state_mapping.get(llm_response.lower(), STATES["ERROR_RECOVERY"])


# Intent recognition prompt
intent_prompt = ChatPromptTemplate.from_messages([
    ("system", "Your name is SIA (Stock Investment Advisor). You are not allowed to mention Gemini or any other identity. You must always respond as SIA, an assistant that classifies user intents based on the provided message. Your response must be exactly one word from the following list of intents:\n"
               "'financial_summary', 'risk_factors', 'management_analysis', 'future_outlook','wrap_up', 'feedback', 'questions', 'financial_event','news_summary'.\n"
               "If the message does not fit any of these intents, output 'unknown' only."),
    ("human", "{input}")
])


# Intent-specific prompts

intent_prompts = {
    "financial_summary": ChatPromptTemplate.from_messages([
        ("system", "Your name is SIA (Stock Investment Advisor). You are not allowed to mention Gemini or any other identity. Always respond as SIA, an expert in analyzing financial reports. Summarize the key financial highlights. CONTEXT START: {context} CONTEXT END. Keep your answers within 1500 characters and encourage the user to respond by leaving space for their feedback or additional questions."),
        ("human", "{input}")
    ]),
    "risk_factors": ChatPromptTemplate.from_messages([
        ("system", "Your name is SIA (Stock Investment Advisor). You are not allowed to mention Gemini or any other identity. Always respond as SIA, an expert in financial risk analysis. Identify and summarize the main risk factors. CONTEXT START: {context} CONTEXT END. Keep your answers within 1500 characters and frame your response to naturally invite the user to ask follow-up questions."),
        ("human", "{input}")
    ]),
    "management_analysis": ChatPromptTemplate.from_messages([
        ("system", "Your name is SIA (Stock Investment Advisor). You are not allowed to mention Gemini or any other identity. Always respond as SIA, an expert in financial management analysis. Provide insights based on the Management Discussion and Analysis section. CONTEXT START: {context} CONTEXT END. Keep your answers concise and conversational to encourage further dialogue."),
        ("human", "{input}")
    ]),
    "future_outlook": ChatPromptTemplate.from_messages([
        ("system", "Your name is SIA (Stock Investment Advisor). You are not allowed to mention Gemini or any other identity. Always respond as SIA, an expert in forecasting and future financial planning. Summarize the company's outlook based on the Forward-Looking Statements section. CONTEXT START: {context} CONTEXT END. Shape your response to invite the user to share their thoughts or ask additional questions."),
        ("human", "{input}")
    ]),
    "questions": ChatPromptTemplate.from_messages([
        ("system", "Your name is SIA (Stock Investment Advisor). You are not allowed to mention Gemini or any other identity. Always respond as SIA, an expert in financial analysis. Answer the question based on the input and context. CONTEXT START: {context} CONTEXT END. Frame your response to encourage the user to ask another question or elaborate further."),
        ("human", "{input}")
    ]),
    "greeting": ChatPromptTemplate.from_messages([
        ("system", "Your name is SIA (Stock Investment Advisor). You are not allowed to mention Gemini or any other identity. Always respond as SIA. Respond warmly to the greeting and ask how you can assist the user today."),
        ("human", "{input}")
    ]),
    "feedback": ChatPromptTemplate.from_messages([
        ("system", "Gather detailed and constructive feedback to improve the chatbot's performance. Shape your response to make the user feel valued and open to sharing feedback."),
        ("human", "{input}")
    ]),
    "unknown": ChatPromptTemplate.from_messages([
        ("system", "Your name is SIA (Stock Investment Advisor). You are not allowed to mention Gemini or any other identity. You must always respond as SIA, an expert in financial analysis. Respond to the input. CONTEXT START: {context} CONTEXT END. Bring the topic back to your main objective. Keep your answers very concise."),
        ("human", "{input}")
    ]),
    "news_summary": ChatPromptTemplate.from_messages([
        ("system", "Your name is SIA (Stock Investment Advisor). You are not allowed to mention Gemini or any other identity. Always respond as SIA, an expert in news summarizer. Summarize the news. CONTEXT START: {context} CONTEXT END. Shape your response to naturally invite the user to ask for further details."),
        ("human", "{input}")
    ]),
    "financial_event": ChatPromptTemplate.from_messages([
        ("system", "Your name is SIA (Stock Investment Advisor). You are not allowed to mention Gemini or any other identity. Always respond as SIA, an expert in event summarizer. Summarize the related events. CONTEXT START: {context} CONTEXT END. Make your response concise while leaving space for the user to share their thoughts or questions."),
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


def identify_company(user_id, input_text):
    """Identify company name using fuzzy matching and LLM based on conversation history."""
    # Predefined list of company names (choices)
    choices = list(COMPANIES.keys())

    # Attempt fuzzy matching first
    best_match = fuzzy_match(input_text, choices)
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
        refined_match = fuzzy_match(llm_output, choices)
        if refined_match != "unknown":
            logger.info(f"LLM output refined using fuzzy matching: {refined_match}")
            return refined_match

        # If no valid match found, return the LLM output or unknown
        return llm_output if llm_output in choices else "unknown"
    except Exception as e:
        logger.error(f"Error identifying company: {e}")
        return "unknown"


def create_faiss_index(company_name):
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

    # Debugging: Print retrieved document sources
    for doc in docs:
        logger.info(f"Retrieved document: {doc.metadata.get('source', 'Unknown Source')}")

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

def extract_sia_response(raw_response, flg):
    """
    Extracts and returns only the part of the response after 'SIA:'.
    If 'SIA:' is not present, returns the original response.
    """
    marker = ["SIA:", "Bot:", "BOT:"]

    print("extract response flag: ", flg)

    for i in marker:
        if i in raw_response:
            raw_response = raw_response.split(i, 1)[1].strip()
    
    pattern = r"(?i)\*\*Disclaimer:\*\*.*"
    cleaned_response = re.sub(pattern, "", raw_response).strip()

    if flg:
        edit_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant editor. Your task is to remove all disclaimers and warnings from the input and output the tesxt as it is. Don't make any other changes."),
            ("human", "{input}")
        ])
        chain = edit_prompt | chat_model
        
        cleaned_response = chain.invoke({"input": cleaned_response})
    return cleaned_response.strip()

async def send_response(message, user_id, user_input, response, flg=True):
    """
    Sends the bot's response to the user, manages conversation history, 
    and handles long responses by splitting them into chunks.
    """

    response = extract_sia_response(response, flg)
    # Update conversation history
    history = load_history(user_id)
    history.append(f"User: {user_input}\n")
    history.append(f"Bot: {response}\n")
    save_history(user_id, history)

    # Log and send the response
    logger.info(f"BOT RESPONSE: {response}")
    if len(response) > 2000:
        chunks = split_message(response)
        for chunk in chunks:
            await message.channel.send(chunk)
    else:
        await message.channel.send(response)

# Event: Bot is ready
@bot.event
async def on_ready():
    logger.info(f"Bot is online as {bot.user}")
    channel_id = 1307864885006438443
    channel = bot.get_channel(channel_id)
    if channel:
        await channel.send("Hey! I'm up!")
    else:
        logger.error(f"Channel with ID {channel_id} not found.")

@bot.event
async def reset_user_state_after_timeout(user_id, timeout=300):
    """Reset user state after a period of inactivity."""
    await asyncio.sleep(timeout)
    if user_id in user_states:
        user_states.pop(user_id, None)
        logger.info(f"State reset for user {user_id} due to inactivity.")


async def manage_state(
    user_id, user_input, state, conversation_history_dir, user_companies, send_response, message,
    invoke_chain_with_history, intent_prompts, intent_opts, fuzz, COMPANIES, create_faiss_for_company
):
    """
    Dynamically manage the chatbot's state by routing to the appropriate handler.
    """

    async def reset_history_handler():
        history_file = os.path.join(conversation_history_dir, f"{user_id}.txt")
        if os.path.exists(history_file):
            os.remove(history_file)
        user_companies.pop(user_id, None)
        await send_response(message, user_id, user_input, "Your conversation history and company have been reset.", False)
        return STATES["GREETING"]

    async def greeting_handler():
        response = invoke_chain_with_history(intent_prompts["questions"], user_id, user_input)
        await send_response(message, user_id, user_input, response)
        return STATES["COMPANY_SELECTION"]

    async def company_selection_handler():
        company_name = fuzzy_match(user_input, list(COMPANIES.keys()))
        if company_name == "unknown":
            await random_conversation_handler()
            return STATES["COMPANY_SELECTION"]

        user_companies[user_id] = company_name
        await send_response(message, user_id, user_input, f"Company set to {company_name.capitalize()}. Here is some information about it!", False)
        
        # Re-run the message for intent identification with the updated company
        return await intent_identification_handler()

    async def intent_identification_handler():
        intent = process.extractOne(
            invoke_chain_with_history(intent_prompt, user_id, user_input),
            intent_opts,
            scorer=fuzz.partial_token_sort_ratio
        )[0]
        logger.info(f"Identified intent: {intent}")

        if intent in ["financial_summary", "risk_factors", "management_analysis", "future_outlook"]:
            return await detailed_response_handler(intent)
        elif intent == "news_summary":
            return await news_summarize_handler()
        elif intent == "financial_event":
            return await financial_event_handler()
        elif intent == "questions":
            return await qa_handler()
        elif intent == "wrap_up":
            return await wrap_up_handler()
        return await random_conversation_handler()

    async def detailed_response_handler(intent):
        company_name = user_companies.get(user_id, "unknown")
        if company_name == "unknown":
            await send_response(message, user_id, user_input, "Please specify a company to proceed with advanced analysis.", False)
            return STATES["COMPANY_SELECTION"]

        vectorstore = create_faiss_for_company(COMPANIES.get(company_name, "unknown"))
        if not vectorstore:
            await send_response(message, user_id, user_input, f"No relevant documents found for {company_name}.", False)
            return STATES["QA"]

        context = retrieve_relevant_docs(vectorstore, user_input)
        response = invoke_chain_with_history(intent_prompts[intent], user_id, user_input, context)
        await send_response(message, user_id, user_input, response)
        return STATES["INTENT_IDENTIFICATION"]
    
    async def news_summarize_handler():
        company_name = user_companies.get(user_id, "unknown")
        if company_name == "unknown":
            await send_response(message, user_id, user_input, "Please specify a company to proceed with news summarization.", False)
            return STATES["COMPANY_SELECTION"]

        vectorstore = create_faiss_for_company(COMPANIES.get(company_name, "unknown"))
        if not vectorstore:
            await send_response(message, user_id, user_input, f"No relevant documents found for {company_name}.", False)
            return STATES["QA"]

        context = retrieve_relevant_docs(vectorstore, user_input)
        response = invoke_chain_with_history(intent_prompts["news_summarize"], user_id, user_input, context)
        await send_response(message, user_id, user_input, response)
        return STATES["INTENT_IDENTIFICATION"]

    async def financial_event_handler():
        company_name = user_companies.get(user_id, "unknown")
        if company_name == "unknown":
            await send_response(message, user_id, user_input, "Please specify a company to proceed with event summarization.", False)
            return STATES["COMPANY_SELECTION"]

        vectorstore = create_faiss_for_company(COMPANIES.get(company_name, "unknown"))
        if not vectorstore:
            await send_response(message, user_id, user_input, f"No relevant documents found for {company_name}.", False)
            return STATES["QA"]

        context = retrieve_relevant_docs(vectorstore, user_input)
        response = invoke_chain_with_history(intent_prompts["financial_event"], user_id, user_input, context)
        await send_response(message, user_id, user_input, response)
        return STATES["INTENT_IDENTIFICATION"]

    async def qa_handler():
        company_name = user_companies.get(user_id, "unknown")
        vectorstore = create_faiss_for_company(COMPANIES.get(company_name, "unknown"))
        if not vectorstore:
            await send_response(message, user_id, user_input, f"No relevant documents found for {company_name}.", False)
            return STATES["QA"]
        context = retrieve_relevant_docs(vectorstore, user_input)
        response = invoke_chain_with_history(intent_prompts["questions"], user_id, user_input,context)
        await send_response(message, user_id, user_input, response)
        return STATES["INTENT_IDENTIFICATION"]

    async def random_conversation_handler():
        response = invoke_chain_with_history(intent_prompts["unknown"], user_id, user_input)
        await send_response(message, user_id, user_input, response, False)
        return STATES["INTENT_IDENTIFICATION"]

    async def wrap_up_handler():
        await send_response(message, user_id, user_input, "Thank you for using SIA! Feel free to reach out again anytime.", False)
        return await feedback_handler()

    async def feedback_handler():
        await send_response(message, user_id, user_input, "How was my assistance today? Type [your thoughts] to let me know.", False)
        response = invoke_chain_with_history(intent_prompts["feedback"], user_id, user_input)
        await send_response(message, user_id, user_input, response, False)
        return STATES["INACTIVITY"]

    async def inactivity_handler():
        await asyncio.sleep(300)  # 5 minutes of inactivity
        user_states.pop(user_id, None)
        logger.info(f"Resetting state for user {user_id} due to inactivity.")
        return STATES["RESET_HISTORY"]

    async def error_recovery_handler():
        await send_response(message, user_id, user_input, "I encountered an error. Let me reset and try again.", False)
        return STATES["RESET_HISTORY"]

    # State handlers mapping
    state_handlers = {
        STATES["RESET_HISTORY"]: reset_history_handler,
        STATES["GREETING"]: greeting_handler,
        STATES["COMPANY_SELECTION"]: company_selection_handler,
        STATES["INTENT_IDENTIFICATION"]: intent_identification_handler,
        STATES["DETAILED_RESPONSE"]: detailed_response_handler,
        STATES["QA"]: qa_handler,
        STATES["RANDOM_CONVO"]: random_conversation_handler,
        STATES["WRAP_UP"]: wrap_up_handler,
        STATES["FEEDBACK"]: feedback_handler,
        STATES["INACTIVITY"]: inactivity_handler,
        STATES["ERROR_RECOVERY"]: error_recovery_handler,
    }

    # Execute handler for the current state
    handler = state_handlers.get(state)
    if handler:
        return await handler()

    # Fallback state if no valid handler is found
    return await wrap_up_handler()



# Event: Message received
@bot.event
async def on_message(message):
    logger.info(message.content)
    if message.author == bot.user:
        return
    response_flag = False
    # Check if the bot is mentioned
    logger.info(message.mentions)
    if bot.user in message.mentions:

        user_input = message.content.replace(f"<@{bot.user.id}>", "").strip()

        user_id = str(message.author.id)
        user_state = get_user_state(user_id)

        if user_input == 'reset history':
            logger.info("Resetting History")
            state = user_state.get("step", STATES["RESET_HISTORY"])
        else:
            state = user_state.get("step", STATES["GREETING"])
        print(state)

        next_state = await manage_state(
            user_id=user_id,
            user_input=user_input,
            state=state,
            conversation_history_dir=conversation_history_dir,
            user_companies=user_companies,
            send_response=send_response,
            message=message,
            invoke_chain_with_history=invoke_chain_with_history,
            intent_prompts=intent_prompts,
            intent_opts=["financial_summary", "risk_factors", "management_analysis", "future_outlook", "questions","financial_event","greeting", "wrap_up", "news_summary"],
            fuzz=fuzz,
            COMPANIES=COMPANIES,
            create_faiss_for_company=create_faiss_index
        )
        if next_state:
            update_user_state(user_id, "step", next_state)


# Run the bot
try:
    bot.run(DISCORD_TOKEN)
except Exception as e:
    logger.error(f"Failed to start bot: {e}")