from utils import get_trulens_recorder, build_basic_rag_index, get_basic_rag_query_engine
from llama_index.llms import OpenAI

def build_basic_rag_engine_recorder(documents: list, index_dir: str, app_id: str):
    """Basic 쿼리 엔진과 평가를 위한 recoder를 반환하는 함수

    Args:
        documents (list): Document reader를 통해 읽어온 document를 한 리스트로 합친 리스트 객체
        index_dir (str): 인덱스가 저장될 디렉토리
        app_id (str): 대시보드 로깅을 위해 쓸 app_id. ex) Basic RAG

    Returns:
        sentence_window_engine: 쿼리엔진
        recorder: 대시보드 로깅을 위한 TruLens recorder
        
    """
    basic_index = build_basic_rag_index(
        documents,
        llm=OpenAI(model="gpt-3.5-turbo", temperature=0.1),
        embed_model="local:BAAI/bge-small-en-v1.5",
        save_dir=index_dir,
    )

    basic_rag_engine = get_basic_rag_query_engine(
        basic_index
    )

    recorder= get_trulens_recorder(
        basic_rag_engine,
        app_id=app_id
    )
    
    return basic_rag_engine, recorder