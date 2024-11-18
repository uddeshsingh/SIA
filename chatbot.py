import os
import discord
from discord.ext import commands
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Retrieve environment variables
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")



# Initialize the Groq chat model
chat_model = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0.4,
    groq_api_key=GROQ_API_KEY
)

# Set up the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "Keep your replies concise and to the point. Be formal! You are a helpful CFA certified investment advisor. Always frame your replies and address yourself accordingly. Your job is to identify the the current status of the mentioned stocks and give sound investment advice. Do not let the user stray from the topic"),
    ("human", "{input}")
])

# Combine the prompt and chat model into a chain
chain = prompt | chat_model

# Initialize Discord bot
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    print(f"Bot is online as {bot.user}")

@bot.event
async def on_message(message):
    print(f"Message from {message.author}: {message.content}")
    if message.author == bot.user:
        return

    if message.content.startswith("!ask"):
        user_input = message.content[5:].strip()
        response = chain.invoke({"input": user_input})
        await message.channel.send(response.content)

    elif message.content.startswith("!hello"):
        user_input = message.content[1:].strip()
        response = chain.invoke({"input": user_input})
        await message.channel.send(response.content)

# Run the bot
bot.run(DISCORD_TOKEN)
