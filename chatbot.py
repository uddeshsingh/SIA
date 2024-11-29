import os
import logging
import discord
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Retrieve environment variables
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DiscordBot")

# Initialize the Groq chat model
chat_model = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0.4,
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

conversation_history = {}

# Intent-specific prompts
intent_prompts = {
    "financial_summary": ChatPromptTemplate.from_messages([
        ("system", "You are an expert in analyzing financial reports. Summarize the key financial highlights from the provided 10k or 10q report."),
        ("human", "{input}")
    ]),
    "risk_factors": ChatPromptTemplate.from_messages([
        ("system", "You are an expert in financial risk analysis. Identify and summarize the main risk factors mentioned in the provided 10k or 10q report."),
        ("human", "{input}")
    ]),
    "management_analysis": ChatPromptTemplate.from_messages([
        ("system", "You are an expert in financial management analysis. Provide insights based on the Management Discussion and Analysis section of the provided report."),
        ("human", "{input}")
    ]),
    "revenue_analysis": ChatPromptTemplate.from_messages([
        ("system", "You are an expert in revenue analysis. Analyze the revenue details from the provided 10k or 10q report and explain any trends or insights."),
        ("human", "{input}")
    ]),
    "expense_analysis": ChatPromptTemplate.from_messages([
        ("system", "You are an expert in financial analysis. Break down and analyze the expense-related information from the provided report."),
        ("human", "{input}")
    ]),
    "liquidity_analysis": ChatPromptTemplate.from_messages([
        ("system", "You are an expert in liquidity and solvency analysis. Review the liquidity-related details from the provided 10k or 10q report and explain the company's financial health."),
        ("human", "{input}")
    ]),
    "future_outlook": ChatPromptTemplate.from_messages([
        ("system", "You are an expert in forecasting and future financial planning. Summarize the company's outlook based on the Forward-Looking Statements section of the provided report."),
        ("human", "{input}")
    ]),
    "competitive_position": ChatPromptTemplate.from_messages([
        ("system", "You are an expert in competitive analysis. Analyze the company's competitive position and market strategy based on the information in the provided report."),
        ("human", "{input}")
    ]),
    "debt_analysis": ChatPromptTemplate.from_messages([
        ("system", "You are a financial analyst specializing in debt. Review the company's debt-related details in the provided report and provide insights on its leverage and repayment capabilities."),
        ("human", "{input}")
    ]),
    "segment_analysis": ChatPromptTemplate.from_messages([
        ("system", "You are an expert in segment analysis. Analyze the performance of different business segments mentioned in the provided report."),
        ("human", "{input}")
    ]),
    "unknown": ChatPromptTemplate.from_messages([
        ("system", "You are an expert in financial analysis. Responsd to the input. If the input is on your expertise, continue the conversation, else bring the topic back to your main objective. Keep your answers very concise"),
        ("human", "{input}")
    ])
}


# Initialize Discord bot
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
bot = discord.Client(intents=intents)

# Helper function for invoking a chain
def invoke_chain_with_history(prompt_template, user_id, input_text):
    # Retrieve or initialize user history
    history = conversation_history.get(user_id, [])
    
    # Combine conversation history with the current input
    full_context = "\n".join(history + [f"User: {input_text}"])
    try:
        chain = prompt_template | chat_model
        response = chain.invoke({"input": full_context})
        
        # Update conversation history
        history.append(f"User: {input_text}")
        history.append(f"Bot: {response.content.strip()}")
        conversation_history[user_id] = history[-10:]  # Keep only the last 10 messages for context
        
        return response.content.strip()
    except Exception as e:
        logger.error(f"Error invoking chain with history: {e}")
        return "Sorry, I couldn't process your request right now."

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
        
        # Handle reset command
        if user_input.lower() == "!reset":
            conversation_history[message.author.id] = []
            await message.channel.send("Your conversation history has been reset.")
            return
        
        # Recognize intent
        intent = invoke_chain_with_history(intent_prompt, message.author.id, user_input)
        logger.info(f"The intent is: {intent}")

        # Respond based on intent
        if intent in intent_prompts:
            response = invoke_chain_with_history(intent_prompts[intent], message.author.id, user_input)
        else:
            response = invoke_chain_with_history(intent_prompts["unknown"], message.author.id, user_input)

        await message.channel.send(response)

# Run the bot
try:
    bot.run(DISCORD_TOKEN)
except Exception as e:
    logger.error(f"Failed to start bot: {e}")