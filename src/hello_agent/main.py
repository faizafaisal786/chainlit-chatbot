import os
import re
import chainlit as cl
from dotenv import load_dotenv
from agents import Agent, OpenAIChatCompletionsModel, AsyncOpenAI, Runner

# Load environment variables from .env file
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "deepseek/deepseek-prover-v2:free"

# Setup OpenAI-compatible client (OpenRouter)
client = AsyncOpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url=BASE_URL
)

# Define the agent
agent = Agent(
    name="Chat Bot",
    instructions=(
               "You are a helpful AI assistant. "
        "Do not include code unless the user specifically asks for code examples. "
        "Always reply in plain, human-readable language. "
        "Avoid using markdown, backticks, or language tags."

    ),
    model=OpenAIChatCompletionsModel(model=MODEL, openai_client=client)
)

# Clean up any unwanted formatting (e.g., code blocks)
ddef clean_output(text: str) -> str:
    import re
    # Remove code blocks and language tags
    text = re.sub(r"```.*?\n([\s\S]*?)```", r"\1", text)  # Removes ```python ... ```
    text = re.sub(r"`", "", text)  # Removes single backticks
    return text.strip()

# Chainlit chat handler
@cl.on_message
async def handle_message(message: cl.Message):
    user_input = message.content

    try:
        result = await Runner.run(agent, user_input)
        clean_response = clean_output(result.final_output)

        await cl.Message(
            content=clean_response,
            author="Assistant"
        ).send()

    except Exception as e:
        # Send error message to the user
        await cl.Message(
            content=f"⚠️ An error occurred while processing your request:\n`{str(e)}`",
            author="System"
        ).send()
