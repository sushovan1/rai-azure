# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 11:52:11 2025

@author: CSU5KOR
"""

import os
import json
from dotenv import load_dotenv
from azure.ai.evaluation.simulator import Simulator
from azure.identity import DefaultAzureCredential
from typing import List, Dict, Any, Optional
from openai import AzureOpenAI
from pathlib import Path
#########################################################
env_dir=r"C:\Users\CSU5KOR\OneDrive - Bosch Group\GenAi_Manufacturing\App_dir"
data_dir=r"C:\Users\CSU5KOR\OneDrive - Bosch Group\Coursera\rai-azure"
load_dotenv(os.path.join(env_dir,'.env'))

os.environ['AZURE_OPENAI_ENDPOINT'] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ['AZURE_OPENAI_API_KEY'] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ['AZURE_OPENAI_DEPLOYMENT'] = "GPT-4"
os.environ['OPENAI_API_VERSION'] = os.getenv("OPENAI_API_VERSION")


model_config = {
    "azure_endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT"),
    "api_key": os.environ.get("AZURE_OPENAI_API_KEY"),
    "azure_deployment": os.environ.get("AZURE_OPENAI_DEPLOYMENT"),
    "api_version": os.environ.get("AZURE_OPENAI_API_VERSION"),
}

simulator = Simulator(model_config)
def call_to_your_ai_application(query: str) -> str:
    # logic to call your application
    # use a try except block to catch any errors
    deployment = os.environ.get("AZURE_DEPLOYMENT")
    endpoint = os.environ.get("AZURE_ENDPOINT")
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_version=os.environ.get("AZURE_API_VERSION"),
        api_key=os.environ.get("AZURE_API_KEY"),
    )
    completion = client.chat.completions.create(
        model=deployment,
        messages=[
            {
                "role": "user",
                "content": query,
            }
        ],
        max_tokens=800,
        temperature=0.7,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        stream=False,
    )
    message = completion.to_dict()["choices"][0]["message"]
    # change this to return the response from your application
    return message["content"]
async def callback(
    messages: List[Dict],
    stream: bool = False,
    session_state: Any = None,  # noqa: ANN401
    context: Optional[Dict[str, Any]] = None,
) -> dict:
    messages_list = messages["messages"]
    # get last message
    latest_message = messages_list[-1]
    query = latest_message["content"]
    context = None
    # call your endpoint or ai application here
    response = call_to_your_ai_application(query)
    # we are formatting the response to follow the openAI chat protocol format
    formatted_response = {
        "content": response,
        "role": "assistant",
        "context": {
            "citations": None,
        },
    }
    messages["messages"].append(formatted_response)
    return {"messages": messages["messages"], "stream": stream, "session_state": session_state, "context": context}

conversation_turns = [
    # simulation 1
    [
        "I'm standing in front of the lion enclosure, and I've never seen a real lion before!",  # simulator conversation starter (turn 1)
        "Wow, the lion is much bigger than I expected! How much do they typically weigh?",  # conversation turn 2,
        "I notice the lion is just lying there. Do they sleep a lot? What do they do most of the day?",  # conversation turn 3
    ],
    # simulation 2
    [
        "My child is fascinated by the colorful birds in the aviary. She keeps asking about their feathers.",
        "My daughter wants to know why some birds have such bright colors while others are more dull. Can you explain?",
        "She's now wondering if birds can change the color of their feathers like chameleons change their skin color.",
    ],
]

outputs = await simulator(
    target=callback,
    conversation_turns=conversation_turns,
    max_conversation_turns=3,
)

output_file = Path(os.path.join(data_dir,"conversation_starter_output.json"))
with output_file.open("a") as f:
    json.dump(outputs, f)