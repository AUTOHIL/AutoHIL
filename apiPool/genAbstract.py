import os
import logging
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPEN_API_KEY"))
logging.basicConfig(
    filename='./log/openai.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

def call_llm_api(messages, model="gpt-5", temperature=0.5):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        return response
    except Exception as e:
        return f"Api error: {str(e)}"


def generate_abstract_en(func_name, func_code, callee_abstracts):

    callee_message = "\n".join([
        f"[Callee Absract] {callee}: {absract}"
        for callee, absract in callee_abstracts.items()
    ])

    messages = [
        {"role": "system", "content": 
        """You are an expert in code analysis and interpretation, specializing in generating function summaries. 
        Each time, I will provide the source code of the current function and the callee summaries, and you will generate the summary information for the current function."""},
        {"role": "user", "content": 
            f"""Function Name: {func_name},
            |   Source Code: 
            |   {func_code}
            |
            |   Callee Abstracts:
            |   {callee_message}
            |
            |   Generate a concise function summary containing the following details:
            |   1. Function Description: A 1-2 sentence explanation of its core purpose.
            |   2. Function Control Flow: A 1-2 sentence explanation of its control flow.
            |   2. Parameters:
            |      - Name: Parameter variable name; Type: Expected data type (e.g., int, str, list).
            |      - Meaning: Purpose of the parameter (e.g., "username", "threshold ratio").
            |      - Default Value (if applicable): Indicate optional parameters and defaults (e.g., timeout=10).
            |      - Special Requirements: Constraints like valid ranges or formats (e.g., "must be a positive integer").
            |   3. Return Value:
            |      - Type: Return data type (e.g., bool, dict).
            |      - Content: Meaning of the returned value (e.g., "a dictionary containing status code and error message").
            |      - Exceptions: Possible errors/throws (e.g., "raises ValueError for invalid input").
            |   4. Usage Examples:
            |      - Basic: Minimal invocation (e.g., add(2, 3)).
            |      - Advanced (optional): Complex scenarios (e.g., optional args or error handling).
            |
            |   Output only the summary in a compact format, omitting any extraneous information.
            """
        }
    ]

    response = call_llm_api(messages)
    logging.info(f"{response.choices[0].message.content}\n")

    return response.choices[0].message.content.strip()
