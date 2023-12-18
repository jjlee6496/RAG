from utils import get_trulens_recorder, build_sentence_window_index, get_sentence_window_query_engine
from llama_index.llms import OpenAI

def build_sentence_window_engine_recorder(documents: list, index_dir: str, window_size: int, app_id: str):
    """Sentence window 쿼리 엔진과 평가를 위한 recoder를 반환하는 함수

    Args:
        documents (list): Document reader를 통해 읽어온 document를 한 리스트로 합친 리스트 객체
        index_dir (str): 인덱스가 저장될 디렉토리
        window_size (int): 윈도우 크기
        app_id (str): 대시보드 로깅을 위해 쓸 app_id. ex)sentence_window 5

    Returns:
        sentence_window_engine: 쿼리엔진
        recorder: 대시보드 로깅을 위한 TruLens recorder
        
    """
    sentence_index = build_sentence_window_index(
        documents=documents,
        llm=OpenAI(model="gpt-3.5-turbo", temperature=0.1),
        embed_model="local:BAAI/bge-small-en-v1.5",
        sentence_window_size=window_size,
        save_dir=index_dir,
    )
    
    sentence_window_engine = get_sentence_window_query_engine(
        sentence_index
    )
    
    recorder = get_trulens_recorder(
        sentence_window_engine,
        app_id=app_id
    )
    return sentence_window_engine, recorder