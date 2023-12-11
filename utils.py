import os
from dotenv import load_dotenv, find_dotenv
import numpy as np

def get_openai_api_key():
    _ = load_dotenv(find_dotenv())

    return os.getenv("OPENAI_API_KEY")


# For Evaluation
from trulens_eval import (
    Feedback,
    TruLlama,
    OpenAI
)

from trulens_eval.feedback import Groundedness

# RAG Triad 로깅을 위한 recorder 생성
def get_trulens_recorder(query_engine, app_id):
    openai = OpenAI()
    
    # Answer Relevance
    qa_relevance = (
        Feedback(openai.relevance_with_cot_reasons, name="Answer Relevance")
        .on_input_output()
        )
    
    # Context Relevance
    qs_relevance = (
        Feedback(openai.relevance_with_cot_reasons, name="Context Relevance")
        .on_input()
        .on(TruLlama.select_source_nodes().node.text)
        .aggregate(np.mean)
    )
    
    # Groundedness
    # grounded = Groundedness(groundedness_provider=openai, summarize_provider=openai)
    grounded = Groundedness(groundedness_provider=openai)
    
    groundedness = (
        Feedback(grounded.groundedness_measure_with_cot_reasons, name="Groundedness")
        .on(TruLlama.select_source_nodes().node.text)
        .on_output()
        .aggregate(grounded.grounded_statements_aggregator)
    )
    
    feedbacks = [qa_relevance, qs_relevance, groundedness]
    tru_recoder = TruLlama(
        query_engine,
        app_id=app_id,
        feedbacks=feedbacks
    )
    return tru_recoder