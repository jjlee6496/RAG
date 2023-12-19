# Setup
import openai
from llama_index import SimpleDirectoryReader
from llama_index import Document
from utils import get_openai_api_key, run_evals

# Evaluation
from query_engines.Sentence_window import build_sentence_window_engine_recorder
from query_engines.RAG_basic import build_basic_rag_engine_recorder
from query_engines.Auto_merging import build_auto_mergning_engine_recorder
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

# Evaluate 할 쿼리엔진 생성        
basic_rag_query_engine, basic_rag_recorder = build_basic_rag_engine_recorder(
                                                        documents=documents,
                                                        index_dir='./index/test1',
                                                        app_id='test_app1'
                                                        )

sentence_window_query_engine, sentence_window_recorder = build_sentence_window_engine_recorder(
                                                          documents=documents,
                                                          index_dir='./index/test2',
                                                          window_size=5,
                                                          app_id='test_app2'
                                                        )

auto_merging_query_engine, auto_merging_recorder = build_auto_mergning_engine_recorder(
                                                    documents=documents,
                                                    chunk_sizes=[2048, 512, 128],
                                                    index_dir='./index/test3',
                                                    app_id='testapp3',
                                                    similarity_top_k=12,
                                                    rerank_top_n=6
                                                    )

# 대시보드 생성
tru = Tru()
tru.reset_database()
# run_evals(eval_questions, basic_rag_recorder, basic_rag_query_engine)
# run_evals(eval_questions, sentence_window_recorder, sentence_window_query_engine)
run_evals(eval_questions, auto_merging_recorder , auto_merging_query_engine)
tru.run_dashboard()
