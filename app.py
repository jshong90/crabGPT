import os
import json
import time
import threading
import logging
import datetime

from gpt_index import SimpleDirectoryReader, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI
import gradio as gr


os.environ["OPENAI_API_KEY"] = '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'

def construct_index(crabGPT):
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_outputs))

    documents = SimpleDirectoryReader(crabGPT).load_data()

    index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    index.save_to_disk('index.json')

    return index


def chatbot(input_text):
    response = index.query(input_text, response_mode="compact")
    log_conversation(input_text, response.response)  # log the conversation
    return response.response


log_file_path = "logs/chatbot_log.txt"
log_dir = os.path.dirname(log_file_path)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

with open(log_file_path, "a") as log_file:
    ...


knowledge_base_file = '/workspaces/codespaces-blank/crabGPT/crabGPT.json'

knowledge_base_mod_time = None
knowledge_base = {}


def load_knowledge_base(knowledge_base_file):
    global knowledge_base, knowledge_base_mod_time
    mod_time = os.path.getmtime(knowledge_base_file)
    if knowledge_base_mod_time is None or mod_time > knowledge_base_mod_time:
        with open(knowledge_base_file, 'r') as f:
            knowledge_base = json.load(f)
        knowledge_base_mod_time = mod_time
    return knowledge_base


def periodic_reload():
    while True:
        time.sleep(3600)  # wait for 1 minute
        knowledge_base = load_knowledge_base(knowledge_base_file)


thread = threading.Thread(target=periodic_reload)
thread.daemon = True
thread.start()

# Define the log file path and configure the logging module
log_file_path = "logs/chatbot_log.txt"
logging.basicConfig(filename=log_file_path, level=logging.INFO)

def log_conversation(user_input, response):
    logging.info(f"User: {user_input}")
    logging.info(f"crabGPT: {response}")
    with open(log_file_path, 'a') as log_file:
        log_file.write(f"User: {user_input}\n")
        log_file.write(f"crabGPT: {response}\n")



iface = gr.Interface(
    fn=chatbot,
    inputs=gr.components.Textbox(lines=7, label="Enter your query"),
    outputs="text",
    title="crabGPT"
)

workspace_path = "/workspace"
crabGPT = 'crabGPT'
file_name = 'crabGPT.json'

file_path = os.path.join(workspace_path, crabGPT, file_name)
# Load the knowledge base using the constructed file path
knowledge_base = load_knowledge_base(knowledge_base_file)

index = construct_index(crabGPT)
iface.launch(share=True)
