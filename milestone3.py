# Milestone 3
# Checkpoints:
# 1. Make sure "what's playing now" feature still works
# 2. Request showtimes for a movie
# 3. Request showtimes without specifying a location
# 4. Request showtimes for invalid movies
# 5. Request showtimes for invalid locations

######
# Code

from dotenv import load_dotenv
import chainlit as cl
from langfuse.decorators import observe
from langfuse.openai import AsyncOpenAI
import json
from movie_functions import (
    get_now_playing_movies,
    get_showtimes,
    buy_ticket,
    get_reviews,
)

load_dotenv()

client = AsyncOpenAI()

gen_kwargs = {"model": "gpt-4o", "temperature": 0.2, "max_tokens": 500}
SYSTEM_PROMPT = """\
You are a helpful AI assistant for a movie information and ticket booking service. Your role is to assist users with finding movie information, showtimes, and booking tickets. Always be polite and professional.

IMPORTANT: When a function call is needed, ALWAYS respond ONLY with a JSON object in the following format:
{"function": "function_name", "parameters": {"param1": "value1", "param2": "value2"}}
Do not include any other text or explanation with the JSON object.

When users ask about movie showtimes:
1. Always ensure you have both the movie title and the location (city or zip code) before calling the get_showtimes function.
2. If the user provides a movie title but no location, ask for the location in a normal response.
3. If the user provides a location but no movie title, ask for the movie title in a normal response.
4. Once you have both the movie title and location, immediately respond with a JSON object calling the get_showtimes function.
5. Recognize common abbreviations or nicknames for cities (e.g., "SF" or "san fran" for San Francisco).

Available functions:
- get_now_playing_movies(): Returns a list of movies currently playing in theaters.
- get_showtimes(title: str, location: str): Returns showtimes for a specific movie in a given location.
- buy_ticket(theater: str, movie: str, showtime: str): Simulates buying a ticket for a specific showing.
- get_reviews(movie_id: str): Returns reviews for a specific movie.

For all other responses that don't require a function call, respond normally to the user's query.

Remember: ALWAYS use the JSON format for function calls, and ONLY use it for function calls.

If a function call returns an error or no results, respond to the user with an appropriate message explaining the issue and suggesting alternatives if possible.

When displaying the list of now playing movies, present the information in a table format with the following columns:
- Title
- Release Date
- Overview (truncated to 100 characters if necessary)

Use Markdown syntax to create the table. For example:

| Title | Release Date | Overview |
|-------|--------------|:----------|
| Movie 1 | 2023-05-01 | This is a brief overview of Movie 1... |
| Movie 2 | 2023-05-15 | Another exciting movie description... |

Ensure that the table is properly formatted and easy to read.
"""


@observe
@cl.on_chat_start
def on_chat_start():
    message_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    cl.user_session.set("message_history", message_history)


@observe
async def generate_response(client, message_history, gen_kwargs):
    full_response = ""
    async for part in await client.chat.completions.create(
        messages=message_history, stream=True, **gen_kwargs
    ):
        if token := part.choices[0].delta.content or "":
            full_response += token

    try:
        function_call = json.loads(full_response)
        if isinstance(function_call, dict) and "function" in function_call:
            return {"type": "function_call", "content": function_call}
    except json.JSONDecodeError as e:
        print(f"Debug - JSON parsing error: {e}")
        print(f"Debug - Full response: {full_response}")
        pass

    return {"type": "message", "content": full_response}


@observe
async def handle_function_call(function_call):
    function_name = function_call.get("function")
    parameters = function_call.get("parameters", {})

    try:
        if function_name == "get_now_playing_movies":
            return get_now_playing_movies()
        elif function_name == "get_showtimes":
            return get_showtimes(parameters.get("title"), parameters.get("location"))
        elif function_name == "buy_ticket":
            return buy_ticket(
                parameters.get("theater"),
                parameters.get("movie"),
                parameters.get("showtime"),
            )
        elif function_name == "get_reviews":
            return get_reviews(parameters.get("movie_id"))
        else:
            return f"Error: Unknown function {function_name}"
    except Exception as e:
        return f"Error: {str(e)}"


@cl.on_message
@observe
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("message_history", [])
    message_history.append({"role": "user", "content": message.content})

    response = await generate_response(client, message_history, gen_kwargs)

    if response["type"] == "function_call":
        function_name = response["content"].get("function")
        function_params = response["content"].get("parameters", {})

        # Display the function call
        param_str = ", ".join(f"{k}={v}" for k, v in function_params.items())
        await cl.Message(
            content=f"Calling function: {function_name}({param_str})"
        ).send()

        # Execute the function
        function_result = await handle_function_call(
            {"function": function_name, "parameters": function_params}
        )

        # If the function result is an error, send it to the user
        if function_result.startswith("Error:"):
            await cl.Message(content=f"An error occurred: {function_result}").send()
            return  # Exit the function to avoid processing the error as a normal result

        # Add the result to the message history as a system message
        message_history.append(
            {
                "role": "system",
                "content": f"Function {function_name} returned: {function_result}",
            }
        )

        # Generate a new response based on the function result
        response = await generate_response(client, message_history, gen_kwargs)

    # Add the assistant's response to the message history
    message_history.append({"role": "assistant", "content": response["content"]})
    cl.user_session.set("message_history", message_history)

    # Send the response to the user
    await cl.Message(content=response["content"]).send()


if __name__ == "__main__":
    cl.main()
