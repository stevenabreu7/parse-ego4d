from datasets import Dataset
from tqdm import tqdm
from utils import load_data, TextDataset
from enum import Enum
from typing import Optional
from pydantic import BaseModel
from openai import OpenAI
import pandas as pd


# model_id  = "gpt-4o"
model_id  = "gpt-4o-2024-08-06"
use_narrations = True
min_correct = 4.0
min_sensible = 4.0
datasplit = "test"

print(f"Model: {model_id}")
print(f"Use narrations: {use_narrations}")
print(f"Min correct: {min_correct}")
print(f"Min sensible: {min_sensible}")
print(f"Data split: {datasplit}")


with open("openai.txt", "r") as f:
    openai_key = f.read().strip()

client = OpenAI(api_key=openai_key)

class ActionCategory(str, Enum):
    mms = "Multimodal search"
    memory = "Memory"
    assistant = "Assistant"
    language = "Language"
    instructions = "Instructions"
    maps = "Maps"
    # other = "Other"

class CoTResponseFormat(BaseModel):
    thoughts: Optional[str]
    action_category: Optional[ActionCategory]

class ResponseFormat(BaseModel):
    action_category: Optional[ActionCategory]


# Load the dataset
df, ua = load_data(min_correct=min_correct, min_sensible=min_sensible, return_map=True)
ds_used = TextDataset(df, tokenizer=None, include_narration=use_narrations, split=datasplit)

def format_query(ds, use_narrations):
    if use_narrations:
        return [f"Narration:\n{n}\nQuery: {q}\n" for q, n in zip(ds.queries, ds.narrations)]
    else:
        return [f"Query: {q}\n" for q in ds.queries]

ds_used = Dataset.from_dict({
    "query": format_query(ds_used, use_narrations),
    "label": ds_used.labels,
})


available_apps = """
- Search: This application will take in the current camera input and a text query \
and run a multimodal search using the text query with the image as context. MMS can \
recognize objects, identify plants and animals, provide nutritional information, \
look up information, and answer general knowledge questions. It takes language and \
image input, and outputs text.
- Assistant_search: This is the Android device assistant that has access to system \
apps. Basic apps that can be called from this app include: `Notes`, `Timer`, \
`Stopwatch`, `Alarm`, `Email`, `Music`, `Phone`, `Contacts`, `Messages`, `Settings`, \
`Calculator`. Additionally, the Assistant_search can control smart home gadgets, \
access notifications, and others.
- Assistant_local: This app can store memories and retrieve them later. Memories can \
be enrolled manually in the app, by the user telling this app to remember something \
explicitly. Memories can also be automatically enrolled without requiring any action \
from the user. For example, if the user is looking at a shopping list, This app \
might automatically remember that shopping list so that it can be retrieved later.
- Language: The language application is an application that can either transcribe \
what the user is hearing right now, translate what the user is reading or hearing, \
determining what language is spoken.
- Directions: The directions application can help the user find relevant places \
nearby, plan routes, estimate distances and navigate to places.
- Assistant_guide: This app can give detailed and step-by-step instructions to the \
user."""
narration_format = "Narration: \n[Narration (up to 200 lines)]"
question_template = f""""You are an intelligent AI assistant living inside of augmented reality (AR) \
glasses. You assist the user in their everyday life with their queries. \
You can call on a number of different applications and actions to fulfil \
the user's queries. In addition to the user's textual query, you will be \
given a textual narration of what the user has been doing most recently \
before asking you their query. Given the narrations and the user's query, \
your task is to decide which application to call on the AR glasses.

The narrations are given in the following format:
(...)
#C C interacts with the man Y
#C C raises a boot
#C C wears the boot on her left leg
#O The man Y walks out of the bedroom
#O The man Y walks into the bedroom
#O The man Y drops the boots on the floor
(...)

where #C shows that the sentence is about an action that the user is doing, \
and #O shows that the sentence is about an action that someone else is doing.

The list of available apps is:
{available_apps}

Given the user query and narration, you should pick the single most relevant \
application to call on the AR glasses.
You will be given the narration, followed by the query, after which you will \
complete your task by calling the application. This will be the format:
{narration_format if use_narrations else ""}
Query: [Query]
Action: [Your response]

Your task begins now.

"""


def get_model_response(model_id: str, sys_prompt: str, user_prompt: str, use_cot: bool = False):
    completion = client.beta.chat.completions.parse(
        model=model_id,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format=CoTResponseFormat if use_cot else ResponseFormat,
    )
    structured_response = completion.choices[0].message.parsed
    return structured_response


results = []
n_correct = 0
n_correct_cot = 0
n_total = 0
it = tqdm(ds_used, total=len(ds_used), desc=f"inference on {datasplit}")
for batch in it:
    question = question_template + batch["query"]
    answers = list(ua)
    label = batch["label"]
    correct_answer = ua[label]

    sr = get_model_response(model_id, question_template.strip(), batch["query"], use_cot=False)
    # cot_sr = get_model_response(model_id, question_template.strip(), batch["query"], use_cot=True)
    try:
        pred_answer = sr.action_category.value
        # cot_pred_answer = cot_sr.action_category.value
    except:
        pred_answer = None
        # cot_pred_answer = None

    correct = correct_answer == pred_answer
    # cot_correct = correct_answer == cot_pred_answer

    n_correct += correct
    # n_correct_cot += cot_correct

    results.append({
        "query": batch["query"].strip().split("\n")[-1],
        "correct_answer": correct_answer,
        "pred_answer": pred_answer,
        # "cot_pred_answer": cot_pred_answer,
        "correct": correct,
        # "cot_correct": cot_correct,
    })
    n_total += 1

    it.set_postfix({
        "accuracy": n_correct/n_total, 
        # "cot_accuracy": n_correct_cot/n_total,
    })

    pd.DataFrame(results).to_csv(f"gpt4o_{use_narrations}_{datasplit}_min{int(min_correct)}_nocot.csv", index=False)
