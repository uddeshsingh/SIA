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
    ("system", "You are an assistant that can classify user intents. Read the full message. It is very crucial you only output the intents given. You will classify messages into intents such as 'stock_query', 'sentiment_analysis', 'information_extraction', or 'general_greeting'. If the message is unrelated to these intents, classify it as 'unknown'. Only output the intents, don't add anything else to the reposnse."),
    ("human", "{input}")
])

# Intent-specific prompts
intent_prompts = {
    "stock_query": ChatPromptTemplate.from_messages([
        ("system", "You are a CFA-certified financial advisor. Answer the user's query about stock performance or investment advice in a concise and formal manner."),
        ("human", "{input}")
    ]),
    "sentiment_analysis": ChatPromptTemplate.from_messages([
        ("system", "You are an expert in sentiment analysis. Analyze the sentiment of the following text and classify it as positive, negative, or neutral."),
        ("human", "{input}")
    ]),
    "information_extraction": ChatPromptTemplate.from_messages([
        ("system", "You are an expert in extracting financial metrics and entities from text."),
        ("human", "{input}")
    ]),
    "general_greeting": ChatPromptTemplate.from_messages([
        ("system", "You are a friendly assistant. Respond politely to general greetings."),
        ("human", "{input}")
    ])
}

# Initialize Discord bot
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
bot = discord.Client(intents=intents)

# Helper function for invoking a chain
def invoke_chain(prompt_template, input_text):
    try:
        chain = prompt_template | chat_model
        response = chain.invoke({"input": input_text})
        return response.content
    except Exception as e:
        logger.error(f"Error invoking chain: {e}")
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
        
        # Step 1: Recognize intent
        intent = invoke_chain(intent_prompt, user_input).strip(' ').strip("'")
        logger.info(f"Recognized intent: {intent}")
        # print("The intent is: ", intent, intent_prompts[intent])
        # Step 2: Respond based on intent
        if intent in intent_prompts:
            response = invoke_chain(intent_prompts[intent], user_input)
        else:
            response = "I'm sorry, I couldn't understand your request. Please provide more context."

        await message.channel.send(response)

# Run the bot
try:
    bot.run(DISCORD_TOKEN)
except Exception as e:
    logger.error(f"Failed to start bot: {e}")
