# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 11:39:26 2025

@author: CSU5KOR
"""

import os
from promptflow.client import load_flow
from dotenv import load_dotenv

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

with open (os.path.join(data_dir,"friendliness.prompty")) as fin:
    print(fin.read())

query = "I've been on hold for 30 minutes just to ask about my luggage! This is ridiculous. Where is my bag?"
response = "I apologize for the long wait time, that must have been frustrating. I understand you're concerned about your luggage. Let me help you locate it right away. Could you please provide your bag tag number or flight details so I can track it for you?"
# response = "Your bag is currently at the airport."

friendliness_eval = load_flow(source=os.path.join(data_dir,"friendliness.prompty"),model={"configuration": model_config})
friendliness_score = friendliness_eval(
    query=query,
    response=response
)

print(friendliness_score)
