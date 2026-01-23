import os
from dotenv import load_dotenv
from typing import List
import logging
import re
import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from the .env file
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def get_deepseek_llm():
    """(LLM) Load and return a DeepSeek LLM instance."""
    api_key = os.getenv("DS_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY environment variable is not set (for LLM)")
    
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=api_key,
        base_url="https://api.deepseek.com/v1"
    )
    return llm

def get_gemini_llm():
    api_key = os.getenv("GOOGLE_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set (for LLM)")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        google_api_key=api_key
    )

    return llm

def bt_encoder(req_decomposed, module_name, requirement_num):
    logging.info("[----------------------- Constructing behavior tree ---------------------]")
    llm = get_gemini_llm()

    template = """[System] You are an expert in constructing Behavior Trees. Based on the API function list provided in the Requirement Analysis (including function names, inputs, outputs, and expected results), please analyze the potential control flow relationships and generate a JSON structure for the Behavior Tree.
The Behavior Tree nodes are defined as follows:
1. Composite Nodes
    -Sequence: Executes all child nodes sequentially. If any child fails, the entire node fails immediately.
    -Selector: Tries each child node sequentially. If any child succeeds, the entire node succeeds immediately.
    -Attributes: type: Node type. name: Node name. children: An array containing nested child nodes.
2. Decorator Nodes
    -Condition: A Boolean check function that controls whether the subsequent child node is executed. attributes: type: Node type. name: Node name. function: (Optional) The specific API function name, if applicable. args: (Optional) Parameters passed to the API, if applicable. params: (Optional) The left-hand side of the conditional expression. expected: (Optional) The remainder of the conditional expression (operator and right-hand side). Must strictly follow Python syntax definitions.
    -Loop: Cyclically executes a child node. attributes: type: Node type. name: Node name. child: The single child node executed in each iteration. count: (Optional) Number of execution cycles; defaults to infinite if omitted. until: (Optional) The exit condition for the loop; must follow Python syntax.
    -ForEach: Iterates over a collection to execute a child node. attributes: type: Node type. name: Node name. child: The single child node executed in each iteration. var: The variable name used to store the current element during each iteration. iterable: The collection to traverse (usually an array or dictionary).
3. Leaf Nodes
    -Action: The specific API function to be executed. Must be selected from the API knowledge base and supports parameter passing.
    -Attributes: type: Node type. name: Node name. function: The name of the API function to execute. args: Parameters passed to the API, represented as key-value pairs.

Constraint Rules
- Variable Definitions: Embed variable definitions within the corresponding Behavior Tree node variables. Do not include undefined attributes in any node.
- Syntax: Conditional logic and expected conditions must strictly adhere to Python syntax definitions (e.g., a null check must be is None).
- Variable Substitution: If the args passed to an API use the variable var (stored from a parent node's iteration), you must encapsulate the parameter value in the format ${args}.
- Data Types & Bases: Pay strict attention to number bases (radix) and data types. You must strictly adhere to the bases and data types specified in the [Requirement Analysis]; modification is prohibited.
- Output: Return only the final Behavior Tree JSON.
[user] Requirement Analysis:
{question}

Answer:
"""

    prompt = ChatPromptTemplate.from_template(template)

    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({
        "question": req_decomposed
    })

    json_string = ""
    # Extract json from markdown code block
    json_match = re.search(r"```json\s*\n(.*?)\n\s*```", answer, re.DOTALL)
    if json_match:
        json_string = json_match.group(1)
    else:
        # If no json block is found, assume the whole response is a json string
        json_string = answer
    
    try:
        bt_json = json.loads(json_string)
        bt_json_dir = f"./reqDecomposer/bt_json/{module_name}"
        os.makedirs(bt_json_dir, exist_ok=True)
        output_path = os.path.join(bt_json_dir, f"{requirement_num}.json")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(bt_json, f, indent=4, ensure_ascii=False)

    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from response: {e}")
        logging.error(f"Problematic string: {json_string}")
        return None
    logging.info("------------------------ Behavior tree construction pipeline finished ----------------------")
    return bt_json, output_path
