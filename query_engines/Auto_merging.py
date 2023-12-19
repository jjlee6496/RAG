from utils import build_automerging_index, get_automerging_query_engine, get_trulens_recorder
from llama_index.llms import OpenAI

def build_auto_mergning_engine_recorder(documents: list, chunk_sizes:list, index_dir: str, app_id: str, similarity_top_k=12, rerank_top_n=6):
    """Auto-merging 쿼리 엔진과 평가를 위한 recoder를 반환하는 함수

    Args:
        documents (list): Document reader를 통해 읽어온 document를 한 리스트로 합친 리스트 객체
        chunk_sizes (list): Hirachical을 구성할 layer수 ex) [2048, 512, 128]
        index_dir (str): 인덱스가 저장될 디렉토리
        app_id (str): 대시보드 로깅을 위해 쓸 app_id. ex) Basic RAG
        similarity_top_k (int): Retrieval 시 사용할 상위 K개
        rerank_top_n (int): Rerank시 사용할 상위 N개

    Returns:
        sentence_window_engine: 쿼리엔진
        recorder: 대시보드 로깅을 위한 TruLens recorder
        
    """
    auto_merging_index = build_automerging_index(
                        documents=documents,
                        llm=OpenAI(model="gpt-3.5-turbo", temperature=0.1),
                        embed_model="local:BAAI/bge-small-en-v1.5",
                        chunk_sizes=chunk_sizes,
                        save_dir=index_dir
                        
    )

    auto_merging_engine = get_automerging_query_engine(
                        automerging_index=auto_merging_index,
                        similarity_top_k=similarity_top_k,
                        rerank_top_n=rerank_top_n
    )

    auto_merging_recorder = get_trulens_recorder(query_engine=auto_merging_engine,
                                                app_id=app_id)
    return auto_merging_engine, auto_merging_recorder