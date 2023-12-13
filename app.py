import openai
from llama_index import SimpleDirectoryReader
from llama_index import Document
from utils import get_openai_api_key, run_evals
from query_engines.Sentence_window import build_sentence_window_engine_recorder
from trulens_eval import Tru

openai.api_key = get_openai_api_key()

# document parsing
documents = SimpleDirectoryReader(
    input_files=["./document/eBook-How-to-Build-a-Career-in-AI.pdf"] # 예시 pdf, 출처:https://info.deeplearning.ai/how-to-build-a-career-in-ai-book
).load_data()

document = Document(text="\n\n".join([doc.text for doc in documents]))

# Evaluation을 위한 질문 불러오기
eval_questions = []
with open('test.text', 'r') as file:
    for line in file:
        # Remove newline character and convert to integer
        item = line.strip()
        eval_questions.append(item)
        
Tru.reset_database()
sentence_window_query_engine, sentence_window_recorder = build_sentence_window_engine_recorder(
                                                        documents=documents,
                                                        index_dir='./index/test',
                                                        window_size=5,
                                                        app_id='test_app'
                                                        )

run_evals(eval_questions, sentence_window_recorder, sentence_window_query_engine)

Tru().run_dashboard()