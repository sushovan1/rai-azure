# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 10:52:23 2025

@author: CSU5KOR
"""

#train_link=https://learn.microsoft.com/en-us/training/modules/run-evaluations-generate-synthetic-datasets/4-risk-safety-metrics
#repo_link=https://github.com/Azure-Samples/RAI-workshops/tree/main/Evaluation%20and%20Data%20Generation%20Workshop
import os
import json
from dotenv import load_dotenv
from azure.ai.evaluation import GroundednessEvaluator,CoherenceEvaluator,FluencyEvaluator,SimilarityEvaluator,F1ScoreEvaluator
from azure.ai.evaluation import RougeScoreEvaluator, RougeType,BleuScoreEvaluator,MeteorScoreEvaluator,GleuScoreEvaluator,RelevanceEvaluator
from azure.ai.evaluation import evaluate
env_dir=r"C:\Users\CSU5KOR\OneDrive - Bosch Group\GenAi_Manufacturing\App_dir"
data_dir=r"C:\Users\CSU5KOR\OneDrive - Bosch Group\Coursera\rai-azure"
load_dotenv(os.path.join(env_dir,'.env'))

os.environ['AZURE_OPENAI_ENDPOINT'] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ['AZURE_OPENAI_API_KEY'] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ['AZURE_OPENAI_DEPLOYMENT'] = "GPT-4"

model_config = {
    "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
    "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
    "azure_deployment":"GPT-4",
}

context = "Dining chair. Wooden seat. Four legs. Backrest. Brown. 18\" wide, 20\" deep, 35\" tall. Holds 250 lbs."
query = "Given the product specification for the Contoso Home Furnishings Dining Chair, provide an engaging marketing product description."
ground_truth = "The dining chair is brown and wooden with four legs and a backrest. The dimensions are 18\" wide, 20\" deep, 35\" tall. The dining chair has a weight capacity of 250 lbs."
response = "Introducing our timeless wooden dining chair, designed for both comfort and durability. Crafted with a solid wood seat and sturdy four-legged base, this chair offers reliable support for up to 250 lbs. The smooth brown finish adds a touch of rustic elegance, while the ergonomically shaped backrest ensures a comfortable dining experience. Measuring 18\" wide, 20\" deep, and 35\" tall, it's the perfect blend of form and function, making it a versatile addition to any dining space. Elevate your home with this beautifully simple yet sophisticated seating option."


groundedness_eval = GroundednessEvaluator(model_config)

groundedness_score = groundedness_eval(
    response=response,
    context=context,
)

print(groundedness_score)

coherence_eval = CoherenceEvaluator(model_config)

coherence_score = coherence_eval(
    response=response,
    query=query
)

print(coherence_score)

fluency_eval = FluencyEvaluator(model_config)

fluency_score = fluency_eval(
    response=response,
    query=query
)

print(fluency_score)

similarity_eval = SimilarityEvaluator(model_config)

similarity_score = similarity_eval(
    response=response,
    query=query,
    ground_truth=ground_truth
)

print(similarity_score)
relevance_eval = RelevanceEvaluator(model_config)

relevance_score = relevance_eval(
    response=response,
    context=context,
    query=query
)

print(relevance_score)
f1_eval = F1ScoreEvaluator()

f1_score = f1_eval(
    response=response,
    ground_truth=ground_truth
)

print(f1_score)
########################################################################
#Quantitative matrix

rouge_eval = RougeScoreEvaluator(rouge_type=RougeType.ROUGE_1)

rouge_score = rouge_eval(
    response=response,
    ground_truth=ground_truth,
)

print(rouge_score)

bleu_eval = BleuScoreEvaluator()

bleu_score = bleu_eval(
    response=response,
    ground_truth=ground_truth
)

print(bleu_score)

meteor_eval = MeteorScoreEvaluator(
    alpha=0.9,
    beta=3.0,
    gamma=0.5
)

meteor_score = meteor_eval(
    response=response,
    ground_truth=ground_truth,
)

print(meteor_score)

gleu_eval = GleuScoreEvaluator()

gleu_score = gleu_eval(
    response=response,
    ground_truth=ground_truth,
)

print(gleu_score)
############################################################
path = os.path.join(data_dir,"performance-quality-data.jsonl")
result = evaluate(
    data=path, # provide your data here
    evaluators={
        "relevance": relevance_eval,
        "groundedness": groundedness_eval,
        "fluency": fluency_eval
    },
    # column mapping
    evaluator_config={
        "default": {
            "query": "${data.query}",
            "response": "${data.response}",
            "context": "${data.context}",
            "ground_truth": "${data.ground_truth}"
        }
    }
)
##########################################################
from pprint import pprint
pprint(result)
import pandas as pd
df=pd.DataFrame(result["rows"])
