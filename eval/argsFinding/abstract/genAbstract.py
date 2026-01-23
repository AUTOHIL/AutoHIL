import os
import logging
import json
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("DS_KEY"), base_url="https://api.deepseek.com")
SUMMARY_CACHE_PATH = "./argsFinding/abstract/summary_cache.json"


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

def call_llm_api(messages, model="deepseek-chat", temperature=0.4):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        return response
    except Exception as e:
        return f"Api error: {str(e)}"


def generate_abstract_en(func_name, func_code):

    messages = [
        {"role": "system", "content": 
        """You are an expert in code analysis and interpretation, specializing in generating function summaries. 
        Each time, I will provide the source code of the current function, the callee summaries and macro definitions, and you will generate the summary information for the current function."""},
        {"role": "user", "content": 
            f"""Function Name: {func_name},
            |   Source Code: 
            |   {func_code}
            |
            |   Generate a concise function summary containing the following details:
            |   1. Function Description: A 1-2 sentence explanation of its core purpose.
            |   2. Function Control Flow: Summarize the overall control flow.
            |   3. Return Value:
            |      - Type: Return data type (e.g., bool, dict).
            |      - Content: Meaning of the returned value (e.g., "a dictionary containing status code and error message").
            |      - Exceptions: Possible errors/throws (e.g., "raises ValueError for invalid input").
            |
            |   Output only the summary in a compact format, omitting any extraneous information.
            """
        }
    ]

    response = call_llm_api(messages)
    logging.info(f"{response.choices[0].message.content}\n")

    return response.choices[0].message.content.strip()


# -- Summary Cache --
def load_summary_cache(path=SUMMARY_CACHE_PATH):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

# -- Save Summary Cache --
def save_summary_cache(summary_map, path=SUMMARY_CACHE_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary_map, f, ensure_ascii=False, indent=2)
    # print(f"Summary cache updated and written to: {path}")


# -- Progress Bar --
def print_progress_bar(current, total, bar_length=30):
    percent = current / total
    filled_length = int(bar_length * percent)
    bar = "█" * filled_length + "░" * (bar_length - filled_length)
    progress_str = f"Progress: [{bar}] {current}/{total} ({int(percent * 100)}%)"
    print(progress_str, end='\r', flush=True)


def main():
    json_map = "./argsFinding/function_info_map.json"

    try:
        with open(json_map, "r") as f:
            function_info_map = json.load(f)

        abstract_map = load_summary_cache()


        total = len(function_info_map)
        processed = len(abstract_map)
        print(f"\n🚀 Starting summary generation, total {total} function nodes ({processed} already cached)")

        for func_name, func_info in function_info_map.items():
            func_code = func_info.get("code", "")
            if not func_code:
                print(f"Warning: No code found for function '{func_name}', skipping.")
                continue

            abstract = generate_abstract_en(func_name, func_code)
            abstract_map[func_name] = abstract
            processed += 1
            print_progress_bar(processed, total)

        save_summary_cache(abstract_map)
        print("Finished.")
    except Exception as e:  
        print(e)

if __name__ == "__main__":
    main()
