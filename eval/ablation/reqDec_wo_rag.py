from dotenv import load_dotenv
import os
import logging
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPEN_API_KEY"))

def requirement_breakdown(requirement_context):
    try:
        messages = [
            {
                "role": "user",
                "content": f"""[Role] You are an expert in Automotive ECU Functional Test Requirement Analysis. For every functional requirement, test scenarios must be simulated to verify if the function operates correctly under those conditions. The Device Under Test (DUT) is the automotive ECU.
[Task] Please strictly search the API knowledge base I have uploaded, identifying and selecting the correct APIs to break down the requirement description into a series of executable steps. Each step must correspond to a found API, a condition, or a jump instruction, adhering to the following requirements:
1. Closed-Loop Testing: The test must form a complete closed loop, including: Prerequisite configuration. Confirmation that the configuration has taken effect. Process verification (specific implementation of the Action requirement). State transitions (ECU Power On/Off). Comparison with expected results (mandatory; if no comparison is needed, skip this). Restoration to a normal state upon test completion (e.g., clearing DTCs).
Note: Process verification may involve repeated testing. Manual testing or undefined custom test steps are strictly prohibited.
2. API Selection & Constraints: Every atomic operation API must be selected strictly and exclusively from the knowledge base I uploaded. You must strictly select APIs for which parameters are explicitly defined in the requirement; do not select APIs if the input parameters are undefined. Parameter values must not be abstract natural language concepts. Prioritize APIs that interact with external ECUs to retrieve information.
Read/Write operation APIs (get/set) must involve interaction with the ECU. DTC retrieval and clearing operations must interact with the external ECU to take effect.
Explicitly state inputs and outputs in the format: input=[], output=[].
3. Control Flow: Mimic machine code conditions/jumps (e.g., jmp) using natural language to describe loops or jumps (e.g., "Repeat steps 2-8").
[Output] Present the complete test step sequence in a table format. The table must include: Step No., Atomic Operation API, Input, Output, Atomic Step Description, and Expected Result. If loops involve custom counters or lists, please provide necessary variable definitions before the control flow description.
Note: Output only the table, the control structure of the requirement, and any necessary variable definitions. 

[Requirement Item]:{requirement_context}
"""
            }
        ]
        
        response = client.chat.completions.create(
            model="gpt-5",
            messages=messages
        )
        logging.info(f"Response: {response.content[0].text}")
        return response.content[0].text

    except Exception as e:
        logging.error(f"Error parsing requirements: {e}")
        return None
    
