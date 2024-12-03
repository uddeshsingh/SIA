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
from langchain.schema import Document
from rapidfuzz import fuzz, process

# Retrieve environment variables
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
VERTEX_AI_API_KEY = os.getenv("VERTEX_AI_API_KEY")

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

STATES = {
    "GREETING": "Greeting and Introduction",
    "RANDOM_CONVO": "Random Conversation",
    "COMPANY_SELECTION": "Company Selection",
    "INTENT_IDENTIFICATION": "Intent Identification",
    "DETAILED_RESPONSE": "Detailed Response",
    "FOLLOW_UP": "Follow-Up Query or Drill-Down",
    "CLARIFICATIONS": "Clarifications or Corrections",
    "SUMMARIZE": "Summarize Conversation Points",
    "NEXT_STEPS": "Call-to-Action or Next Steps",
    "WRAP_UP": "Session Wrap-Up",
    "INACTIVITY": "Inactivity Handling",
    "ERROR_RECOVERY": "Error Recovery",
    "FEEDBACK": "Feedback Loop",
    "QA" : "Questions",
    "RESET_HISTORY" : "Reset the conversation history"
}

id_company = "unknown"
user_states = {}
user_feedback = {}

conversation_history_dir = "user_histories"
os.makedirs(conversation_history_dir, exist_ok=True)

user_companies = {}

# Initialize the Groq chat model
chat_model = VertexAI(
    model="gemini-1.0-pro-002",
    temperature=0.1,
    max_output_tokens=512,
    top_p=0.1
)

def get_user_state(user_id):
    """Retrieve or initialize the user's state."""
    if user_id not in user_states:
        user_states[user_id] = {"step": "initial", "company": None, "intent": None}
    return user_states[user_id]

def update_user_state(user_id, key, value):
    """Update a specific key in the user's state."""
    if user_id in user_states:
        user_states[user_id][key] = value

def save_feedback(user_id, feedback):
    """Save feedback from users."""
    if user_id not in user_feedback:
        user_feedback[user_id] = []
    user_feedback[user_id].append(feedback)
    logger.info(f"Feedback received from user {user_id}: {feedback}")

def get_best_match(company_name, choices):
    # Perform fuzzy matching
    result = process.extractOne(company_name.lower(), choices, scorer=fuzz.partial_token_sort_ratio)
    best_match, score = result[0], result[1]
    return best_match if score > 80 else "unknown"

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
        "clarifications": STATES["CLARIFICATIONS"],
        "summarize": STATES["SUMMARIZE"],
        "next_steps": STATES["NEXT_STEPS"],
        "wrap_up": STATES["WRAP_UP"],
        "inactivity": STATES["INACTIVITY"],
        "error_recovery": STATES["ERROR_RECOVERY"],
        "reset_history": STATES["RESET_HISTORY"],
        "feedback": STATES["FEEDBACK"],
        "questions": STATES["QA"]
    }
    return state_mapping.get(llm_response.lower(), STATES["ERROR_RECOVERY"])


# Intent recognition prompt
intent_prompt = ChatPromptTemplate.from_messages([
    ("system", "Your name is SIA (Stock Investment Advisor). You are not allowed to mention Gemini or any other identity. You must always respond as SIA, an assistant that classifies user intents based on the provided message. Your response must be exactly one word from the following list of intents:\n"
               "'financial_summary', 'risk_factors', 'management_analysis', 'future_outlook', 'clarifications','wrap_up','follow_up', 'feedback', 'questions'.\n"
               "If the message does not fit any of these intents, output 'unknown' only."),
    ("human", "{input}")
])


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
    "future_outlook": ChatPromptTemplate.from_messages([
        ("system", "Your name is SIA (Stock Investment Advisor). You are not allowed to mention Gemini or any other identity. You must always respond as SIA, an expert in forecasting and future financial planning. Summarize the company's outlook based on the Forward-Looking Statements section. CONTEXT START: {context} CONTEXT END. Keep your answers within 1500 characters"),
        ("human", "{input}")
    ]),
    "unknown": ChatPromptTemplate.from_messages([
        ("system", "Your name is SIA (Stock Investment Advisor). You are not allowed to mention Gemini or any other identity. You must always respond as SIA, an expert in financial analysis. Respond to the input. CONTEXT START: {context} CONTEXT END. Bring the topic back to your main objective. Keep your answers very concise."),
        ("human", "{input}")
    ]),
    "greeting": ChatPromptTemplate.from_messages([
        ("system", "Your name is SIA (Stock Investment Advisor). You are not allowed to mention Gemini or any other identity. You must always respond as SIA. Respond to the greeting as SIA, not as a bot."),
        ("human", "{input}")
    ]),
    "follow_up": ChatPromptTemplate.from_messages([
        ("system", "Ask follow-up questions or check if the user needs additional assistance."),
        ("human", "{input}")
    ]),
    "clarifications": ChatPromptTemplate.from_messages([
        ("system", "Ask the user for clarifications to better understand their query."),
        ("human", "{input}")
    ]),
    "intent": ChatPromptTemplate.from_messages([
        ("system", "Clarify the user's intent."),
        ("human", "{input}")
    ]),
    "questions": ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions in an appropriate manner. The length of the anwwer depends on you. ONLY RESPOND WITH THE ANSWER"),
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
        loader = DirectoryLoader(company_folder, glob="**/*.txt")
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
    if not main_doc_included:
        try:
            with open(main_doc_path, "r") as file:
                main_doc_content = file.read()
            main_doc = Document(page_content=main_doc_content, metadata={"source": main_doc_path})
            docs.insert(0, main_doc)  # Add the main document to the top of the results
        except Exception as e:
            print(f"Error loading main document: {e}")

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

def extract_sia_response(raw_response):
    """
    Extracts and returns only the part of the response after 'SIA:'.
    If 'SIA:' is not present, returns the original response.
    """
    marker = ["SIA:", "Bot:", "BOT:"]

    for i in marker:
        if i in raw_response:
            raw_response = raw_response.split(i, 1)[1].strip()
    
    pattern = r"(?i)\*\*Disclaimer:\*\*.*"
    cleaned_response = re.sub(pattern, "", raw_response).strip()
    return cleaned_response.strip()

async def send_response(message, user_id, user_input, response):
    """
    Sends the bot's response to the user, manages conversation history, 
    and handles long responses by splitting them into chunks.
    """

    response = extract_sia_response(response)
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

# Event: Message received
@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    response_flag = False
    # Check if the bot is mentioned
    if bot.user in message.mentions:
        user_input = message.content.replace(f"<@{bot.user.id}>", "").strip()

        user_id = str(message.author.id)
        state = get_user_state(user_id)
        print(state)

        # LLM invocation to decide state transition
        decision_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are SIA (Stock Investment Advisor). Based on the conversation context, decide the next step in the flow. Return exactly one of the following steps: "
                    "greeting, random_conversation, reset_history, intent_identification, detailed_response, questions, summarize, next_steps, wrap_up, inactivity, feedback. Your response must be one of the aforementioned ONLY. BE VERY CONCISE"),
            ("human", f"Current State: {state['step']}\nUser Input: {user_input}")
        ])

        VALID_STATES = ["greeting", "random_conversation", "reset_history", "company_selection","intent_identification", "detailed_response", "questions","next_steps", "wrap_up", "inactivity", "feedback"]
        intent_opts = ["financial_summary",'risk_factors','management_analysis','future_outlook', 'questions','clarifications']

        try:
            decision_chain = decision_prompt | chat_model
            next_state = get_best_match(decision_chain.invoke({"input": user_input}),VALID_STATES)
            
            print(next_state)

            next_state_name = get_next_state(next_state)
            update_user_state(user_id, "step", next_state_name)
            logger.info(f"State changed to: {next_state_name}")
            print('here')

        except Exception as e:
            print('here2')
            logger.error(f"Error determining next state: {e}")
            update_user_state(user_id, "step", STATES["ERROR_RECOVERY"])
            next_state_name = STATES["ERROR_RECOVERY"]
            return
        

        identified_company = identify_company(user_id, user_input)
        logger.info(f"Identified company: {identified_company}")
        if identified_company != "unknown":
            user_companies[user_id] = identified_company

        
        if user_input.lower() == "reset history" or next_state_name == STATES["RESET_HISTORY"]:
            # Reset conversation history and company
            history_file = os.path.join(conversation_history_dir, f"{user_id}.txt")
            if os.path.exists(history_file):
                os.remove(history_file)
            user_companies.pop(user_id, None)
            await send_response(message, user_id, user_input, "Your conversation history and company have been reset.")
            update_user_state(user_id, "step", STATES["GREETING"])
            return
            
        elif next_state_name == STATES["GREETING"]:
            response = invoke_chain_with_history(intent_prompts["greeting"], user_id, user_input)
            await send_response(message, user_id, user_input, response)
            update_user_state(user_id, "step",STATES["COMPANY_SELECTION"])

        elif next_state_name == STATES["RANDOM_CONVO"]:
            response = invoke_chain_with_history(intent_prompts["unknown"], user_id, user_input)
            update_user_state(user_id, "step",STATES["COMPANY_SELECTION"])
            await send_response(message, user_id, user_input, response)

        elif next_state_name == STATES["INTENT_IDENTIFICATION"] or get_user_state(user_id)['step'] == STATES["INTENT_IDENTIFICATION"]:
            # Recognize intent
            intent = process.extractOne(invoke_chain_with_history(intent_prompt, message.author.id, user_input), intent_opts, scorer=fuzz.partial_token_sort_ratio)[0]
            logger.info(f"The intent is: {intent}")
            company_name = user_companies.get(user_id, "unknown")
            if company_name == 'unknown' and intent != 'unknown':
                await send_response(message, user_id, user_input, "I don't think we confirmed the company to discuss yet")
                return
            # intent_chain = intent_prompts['intent'] | chat_model
            # response = intent_chain.invoke({"input":"The current understanding of the intent of the user is: " + intent + "and the company is:" + company_name})
            # print('I am here checking the response of intent', response)
            # await send_response(message, user_id, user_input, response)
            update_user_state(user_id, "intent",intent)
            if intent == 'financial_summary' or intent == 'risk_factors' or intent == 'management_analysis' or intent == 'future_outlook':
                print('in here: changed state to detailed response')
                update_user_state(user_id, "step",STATES["DETAILED_RESPONSE"])
            elif intent == 'questions':
                update_user_state(user_id, "step",STATES["QA"])
                print('in here: changed state to QA')
            elif intent == 'clarifications':
                update_user_state(user_id, "step",STATES["CLARIFICATIONS"])
                print('in here: changed state to CLARIFICATIONS')


        just_dr = False
        if next_state_name == STATES["DETAILED_RESPONSE"] or get_user_state(user_id)['step'] == STATES["DETAILED_RESPONSE"]:
            logger.info('In Detailed Response')
            intent = get_user_state(user_id)["intent"]
            if intent == None:
                intent = process.extractOne(invoke_chain_with_history(intent_prompt, message.author.id, user_input), intent_opts, scorer=fuzz.partial_token_sort_ratio)[0]

            company_name = user_companies.get(user_id, "unknown")

            print("Detailed Response", company_name, intent)
            if company_name != "unknown":
                id_company_ticker = tnc.get(company_name, "unknown")
                vectorstore = create_faiss_for_company(id_company_ticker)
                if not vectorstore:
                    await send_response(message, user_id, user_input, f"No relevant documents found for {company_name}.")
                    return
                context = retrieve_relevant_docs(vectorstore, user_input)
                response = invoke_chain_with_history(intent_prompts[intent], user_id, user_input, context)
                update_user_state(user_id, "step",STATES["QA"])
                just_dr = True
                await send_response(message, user_id, user_input, response)
            else:
                generic_response = "I'm here to help with general questions. For advanced analysis, please set a company by saying something like 'Change company to Tesla'."
                try:
                    print("\n\n\n this is in else of DR")
                    response = invoke_chain_with_history(intent_prompts[intent], user_id, user_input,"Your name is SIA (Stock Investment Advisor). You are not allowed to mention Gemini or any other identity. You must always respond as SIA. You were developed to help users understand the current landscape of top tech companies" )
                    print("\n\n\n this response was created in else of DR", response)
                    logger.info(f"BOT RESPONSE (unknown): {response}")
                    await send_response(message, user_id, user_input, response)
                except Exception as e:
                    logger.error(f"Error generating generic response: {e}")
                    response = generic_response
                    await send_response(message, user_id, user_input, response)
                    return
        
        
        if next_state_name == STATES["QA"] or get_user_state(user_id)['step'] == STATES["QA"]:
            logger.info('In QA')
            if just_dr:
                await send_response(message, user_id, user_input,"Do you have any questions on the above?")
                just_dr = False
                return
            
            response = invoke_chain_with_history(intent_prompts["questions"], user_id, user_input)
            await send_response(message, user_id, user_input, response)
            update_user_state(user_id, "step",STATES["WRAP_UP"])

        if next_state_name == STATES["CLARIFICATIONS"] or get_user_state(user_id)['step'] == STATES["CLARIFICATIONS"]:
            logger.info('In Clarifications')
            await send_response(message, user_id, user_input, "Could you clarify your question so I can provide a more accurate response?")
            return

        if next_state_name == STATES["WRAP_UP"] or get_user_state(user_id)['step'] == STATES["WRAP_UP"]:
            logger.info('In WRAP UP')
            logger.info('In Summarize')
            history = load_history(user_id)
            summary_prompt = ChatPromptTemplate.from_messages([
                ("system", "Summarize the conversation so far for the user."),
                ("human", "\n".join(history))
            ])
            summary_chain = summary_prompt | chat_model
            response = summary_chain.invoke({})
            await send_response(message, user_id, user_input, response)
            await send_response(message, user_id, user_input, "Thank you for using SIA! Feel free to reach out again anytime.")
            update_user_state(user_id, "step",STATES["FEEDBACK"])
        
        inact = True
        if next_state_name == STATES["FEEDBACK"] or get_user_state(user_id)['step'] == STATES["FEEDBACK"]:
            await send_response(message, user_id, user_input, "How was my assistance today? Type [your thoughts] to let me know.")
            response = invoke_chain_with_history(intent_prompts["unknown"],user_id,user_input, "The input is user feedback on your performance as SIA")
            await send_response(message, user_id, user_input, response)
            update_user_state(user_id, "step",STATES["INACTIVITY"])
            inact = False
    
        if next_state_name == STATES["INACTIVITY"] or get_user_state(user_id)['step'] == STATES["INACTIVITY"]:
            if inact:
                await reset_user_state_after_timeout(user_id)
                await send_response(message, user_id, user_input, "It seems you've been inactive. Resetting the session.")
            else:
                await reset_user_state_after_timeout(user_id, 1)
                await send_response(message, user_id, user_input, "Ending the session.")


            


        return


# Run the bot
try:
    bot.run(DISCORD_TOKEN)
except Exception as e:
    logger.error(f"Failed to start bot: {e}")