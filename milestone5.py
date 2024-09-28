# Milestone 5
# Checkpoints:
# - Test the previous features and make sure they still work
# - Try buying a ticket with inadequate parameters
# - Try buying a ticket with positive confirmation
# - Try buying a ticket, but cancel the purchase

######
# Code

from dotenv import load_dotenv
import chainlit as cl
from langfuse.decorators import observe
from langfuse.openai import AsyncOpenAI
import json
import re
from movie_functions import (
    get_now_playing_movies,
    get_showtimes,
    buy_ticket,
    get_reviews,
)

load_dotenv()

client = AsyncOpenAI()

gen_kwargs = {"model": "gpt-4o-mini", "temperature": 0.2, "max_tokens": 500}
SYSTEM_PROMPT = """\
You are a helpful AI assistant for a movie information and ticket booking service. Your role is to assist users with finding movie information, showtimes, and booking tickets. Always be polite and professional.

IMPORTANT: When a function call is needed, ALWAYS respond in the following format:
[FUNCTION_CALL]function_name(param1, param2)[/FUNCTION_CALL]

When users ask about movie showtimes:
1. Always ensure you have both the movie title and the location (city or zip code) before calling the get_showtimes function.
2. If the user provides a movie title but no location, ask for the location in a normal response.
3. If the user provides a location but no movie title, ask for the movie title in a normal response.
4. Once you have both the movie title and location, immediately respond with a function call to get_showtimes.
5. Recognize common abbreviations or nicknames for cities (e.g., "SF" or "san fran" for San Francisco).

Available functions:
- get_now_playing_movies(): Returns a list of movies currently playing in theaters.
- get_showtimes(title, location): Returns showtimes for a specific movie in a given location.
- confirm_ticket_purchase(theater, movie, showtime): Initiates the ticket purchase process and returns a confirmation message.
- get_reviews(movie_id): Returns reviews for a specific movie.

IMPORTANT: When a user wants to buy a ticket, ALWAYS call confirm_ticket_purchase first. NEVER call buy_ticket directly.

The buying process should follow these steps:
1. Call confirm_ticket_purchase with the theater, movie, and showtime.
2. Wait for the user's confirmation (they will type "BUY" to confirm).
3. The system will handle the actual purchase if the user confirms.

For all other responses that don't require a function call, respond normally to the user's query.

Remember: ALWAYS use the [FUNCTION_CALL] format for function calls, and ONLY use it for function calls.

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

    # Check if the response contains a function call
    function_call_match = re.search(
        r"\[FUNCTION_CALL\](.*?)\[/FUNCTION_CALL\]", full_response, re.DOTALL
    )
    if function_call_match:
        function_call = function_call_match.group(1).strip()
        function_name, params_str = function_call.split("(", 1)
        params_str = params_str.rstrip(")")
        params = [param.strip() for param in params_str.split(",") if param.strip()]
        return {
            "type": "function_call",
            "content": {"function": function_name, "parameters": params},
        }

    # If no function call is detected, return the full response as a message
    return {"type": "message", "content": full_response}


@observe
async def handle_function_call(function_call):
    function_name = function_call.get("function")
    parameters = function_call.get("parameters", [])

    try:
        if function_name == "get_now_playing_movies":
            return get_now_playing_movies()
        elif function_name == "get_showtimes":
            return get_showtimes(*parameters)
        elif function_name == "buy_ticket":
            return buy_ticket(*parameters)
        elif function_name == "get_reviews":
            return get_reviews(*parameters)
        elif function_name == "confirm_ticket_purchase":
            return confirm_ticket_purchase(*parameters)
        else:
            return f"Error: Unknown function {function_name}"
    except Exception as e:
        return f"Error: {str(e)}"


@cl.on_message
@observe
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("message_history", [])
    message_history.append({"role": "user", "content": message.content})

    # Check if we're waiting for a purchase confirmation
    awaiting_confirmation = cl.user_session.get("awaiting_confirmation", False)
    if awaiting_confirmation:
        if message.content.lower() == "buy":
            # Proceed with the purchase
            purchase_details = cl.user_session.get("purchase_details")
            result = await handle_function_call(
                {"function": "buy_ticket", "parameters": purchase_details}
            )
            await cl.Message(content=f"Ticket purchased successfully: {result}").send()
        else:
            await cl.Message(content="Purchase cancelled.").send()

        # Reset the confirmation state
        cl.user_session.set("awaiting_confirmation", False)
        cl.user_session.set("purchase_details", None)
        return

    while True:
        response = await generate_response(client, message_history, gen_kwargs)

        if response["type"] == "function_call":
            function_name = response["content"].get("function")
            function_params = response["content"].get("parameters", [])

            # Display the function call
            param_str = ", ".join(function_params)
            await cl.Message(
                content=f"Calling function: {function_name}({param_str})"
            ).send()

            # Execute the function
            function_result = await handle_function_call(response["content"])

            # If the function result is an error, send it to the user
            if isinstance(function_result, str) and function_result.startswith(
                "Error:"
            ):
                await cl.Message(content=f"An error occurred: {function_result}").send()
                return

            # Handle confirm_ticket_purchase separately
            if function_name == "confirm_ticket_purchase":
                cl.user_session.set("awaiting_confirmation", True)
                cl.user_session.set("purchase_details", function_params)
                await cl.Message(content=function_result).send()
                return

            # Add the result to the message history as a system message
            message_history.append(
                {
                    "role": "system",
                    "content": f"Function {function_name} returned: {json.dumps(function_result)}",
                }
            )

            # Generate a new response based on the function result
            continue
        else:
            # Add the assistant's response to the message history
            message_history.append(
                {"role": "assistant", "content": response["content"]}
            )
            cl.user_session.set("message_history", message_history)

            # Send the response to the user
            await cl.Message(content=response["content"]).send()
            break


def confirm_ticket_purchase(theater, movie, showtime):
    confirmation_message = f"You're about to purchase a ticket for '{movie}' at {theater} for {showtime}. Please type BUY to confirm or any other message to cancel."
    return confirmation_message
