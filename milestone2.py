# Milestone 2
# Learnings:
# 1. Design system prompt. Instruct LLM to detect when the user is requesting a list of current movies.
#   If appropriate, generate a function call; otherwise, respond to user. Determine the format of the function call so that it's easy to parse.
# 2. Parse LLM output. If LLM is generating a function call, call it, and inject result as a system message into history.
# 3. If function call, generate additional chat completion request. This makes LLM review function call result and respond to user.
# Checkpoints:
# 1. Ask bot normal movie questions -> should not trigger function call
# 2. Ask bot what's playing -> should trigger function call
# 3. Ask bot for showtimes for a movie -> should trigger function call
# 4. Observe what happens if movie function call fails with an error


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


@cl.on_message
@observe
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("message_history", [])
    message_history.append({"role": "user", "content": message.content})

    response = await generate_response(client, message_history, gen_kwargs)
    print("Debug - Response object:", response)  # Debug output

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
