import openai
import os
from dotenv import load_dotenv
from trulens_eval import Tru

load_dotenv()
openai.api_key = os.environ.get('OPENAI_API_KEY')


# Evaluation

eval_questions = []
with open('generated_questions.text', 'r') as file:
    for line in file:
        # Remove newline character and convert to integer
        item = line.strip()
        eval_questions.append(item)


def run_evals(eval_questions, tru_recorder, query_engine):
    for question in eval_questions:
        with tru_recorder as recording:
            response = query_engine.query(question)

# run_evals(eval_questions, tru_recorder, auto_merging_engine_0)