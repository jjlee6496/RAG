# RAG
- 기본 RAG 방법의 성능을 개선하기 위해 Sentece-window, Auto-merging등 여러가지 방법에 대해서 [Trulens](https://github.com/truera/trulens)를 이용하여 비교하는 프로젝트
## Basic RAG
- 사용자에게 보다 정확한 Response를 제공하기 위해 추가적인 private 또는 real-time data 같은 추가적인 context를 제공하고, 이를 기반으로 response를 생성하는 방법이다.
- 보통 문서를 불러와서 split하여 chunk를 생성, 이를 임베딩하여 vector database에 저장한 후 retrieve하여 사용자 프롬프트(쿼리)와 유사한 context를 제공하여 응답을 생성하는 구조이다.

<table><tr><td><img src='https://python.langchain.com/assets/images/rag_indexing-8160f90a90a33253d0154659cf7d453f.png' width="4200"></td><td><img src='https://python.langchain.com/assets/images/rag_retrieval_generation-1046a4668d6bb08786ef73c56d4f228a.png' width="4200"></td></tr></table>  

- 하지만 retrieve할 때 chunk로 쪼개진 정보를 그대로 사용하여 충분한 context를 제공하지 못한다는 문제점이 있다.

# RAG Triad
![RAG Triad](https://github.com/jjlee6496/RAG/blob/main/imgs/readme/RAG_Triad.png)
- RAG 성능을 평가하기 위해 다음 3가지 metric을 사용한다. 이를 RAG Triad라고 부른다.
## [Anaswer Relevance](https://github.com/jjlee6496/RAG/blob/main/utils.py#L176)
- Output Response가 user query와 얼마나 relevant한지 LLM의 CoT(Chain of Thoughts)로 판단한다.
- User query와 response 사이의 [QuestionStatementRelevance](https://github.com/truera/trulens/blob/21d3dcf6b7ef7ec7b0a7774ac6dbfc6bbd85b86b/trulens_eval/trulens_eval/feedback/v2/feedback.py#L208)를 계산하여 평가
## [Context Relevance](https://github.com/jjlee6496/RAG/blob/main/utils.py#L182)
- Context(retrieved chunks)가 query에 얼마나 relevant한지 LLM의 CoT(Chain of Thoughts)로 판단한다.
- User query와 각각의 context사이의 [QuestionStatementRelevance](https://github.com/truera/trulens/blob/21d3dcf6b7ef7ec7b0a7774ac6dbfc6bbd85b86b/trulens_eval/trulens_eval/feedback/v2/feedback.py#L208)를 계산하여 평균을 계산하여 평가
## [Groundedness](https://github.com/jjlee6496/RAG/blob/main/utils.py#L193)
- Output response가 context(retrieved chunks)에 얼마나 기반하는지에 LLM의 CoT(Chain of Thoughts)로 response와 context의 information overlap에 대하여 평가.
- Retrieval step에서 충분히 많은 관련된 context를 찾지 못한다면 context에 드러난 정보들을 기반으로 답변하는 것이 아닌 pre-trained information을 사용하게 된다. 이는 Groundedness를 떨어뜨린다.
- [Groundedness 코드 부분](https://github.com/truera/trulens/blob/21d3dcf6b7ef7ec7b0a7774ac6dbfc6bbd85b86b/trulens_eval/trulens_eval/feedback/v2/feedback.py#L191)

# Sentence-window
- Sentence-window 기법은 기존 RAG의 문제점인 retrieve시 chunk로 쪼개진 정보를 그대로 사용하여 충분한 context를 제공하지 못한다는 문제점을 해결하기 위한 기법이다.
![Sentence_window_Retrieval](https://github.com/jjlee6496/RAG/blob/main/imgs/readme/Sentence_window_Retrieval.png)
- Retrieve된 context(잘려진 정보)를 metadata로 window size 만큼의 주변 문맥을 포함하도록 대체해 줌으로써 풍부한 context 정보를 얻게 된다.
- [블로그 설명 글](https://velog.io/@jjlee6496/Building-and-Evaluating-Advanced-RAG-1#1-sentence-window-parsing)
## Result
- Sentence-window 기법의 window 크기를 1, 3, 5로 늘려가면서 RAG Triad에 대한 성능 평가
![실험결과1](https://github.com/jjlee6496/RAG/blob/main/imgs/test/sentence_window_comparison.png)
- window size가 1에서 3이 되었을 때 Answer Relevance와 Context Relevance는 하락했지만 Groundedness는 상승했다.
- window size가 5일때 Groundedness와 Context Relevance모두 상승했다.

# Auto-merging
- 기존 RAG는 Retrieve시 많은 chunks를 살펴보는데, 이때 chunking이 잘 되지 않았다면 살펴보는 횟수도 많아지고, 중복된 맥락을 볼 가능성이 높아진다. 이를 방지하기 위해 hierachical을 사용한다.
![Auto_merging_Retrieval](https://github.com/jjlee6496/RAG/blob/main/imgs/readme/Auto_merging_Retrieval.png)  
- Auto-merging 기법은 [HierarchicalNodeParser](https://github.com/run-llama/llama_index/blob/main/llama_index/node_parser/relational/hierarchical.py#L43)를 사용하여 문서를 계층화하여 사용한다.
- 계층화된 관계성을 살펴보며 일정[Threshold](https://github.com/run-llama/llama_index/blob/main/llama_index/retrievers/auto_merging_retriever.py#L12)를 넘어서면, 즉 현재 처리하는 노드의 자식노드의 수가 전체 자식 노드수에서 threshold보다 높다면 이를 하나의 맥락으로 판단한다. 따라서 child node를 parent노드와 합치면서 context사이즈를 늘려 보강해준다.
- [블로그 설명 글](https://velog.io/@jjlee6496/Building-and-Evaluating-Advanced-RAG-2)
## Result
![실험결과2](https://github.com/jjlee6496/RAG/blob/main/imgs/test/auto_merging_comparison.png)
- layer를 2에서 3으로 늘렸을 때 Groundedness가 상승했지만 Answer Relevance와 Context Relevance는 하락했다
- layer를 4로 늘렸을 때 3개지표 모두 개선은 없었다.

# 전체 결과

- Auto Merging기법이 Sentence window기법보다 많은 토큰을 사용했다.
- Auto Merging기법은 Groundedness가 전체적으로 높지만 Context Relevance는 매우 낮았다.
- Sentene window기법(window size가 5 일때)이 비용, RAG Triad 면에서 가장 좋다고 생각된다.

# Reference
- https://python.langchain.com/docs/use_cases/question_answering/
- https://learn.deeplearning.ai/building-evaluating-advanced-rag/lesson/1/lesson_1

# To-do
- [X] RAG 개념 정리
- [X] Basic RAG 코드 작성
- [X] RAG Triad 개념 정리
- [X] Sentence-window Retrieval 개념 정리
- [X] Sentence-window Retrieval 코드 작성
- [X] Auto-merging Retrieval 개념 정리
- [X] Auto-merging Retrieval 코드 작성
- [X] Evaluation 코드 작성
- [ ] 전체 비교 실험 결과 추가
- [ ] 한국어 문서 비교 실험 진행

# 한계점
- LLM을 사용해서 평가하는 만큼 편리하지만 비용이 발생한다.
- 비용을 고려하여 5개의 문장만 평가했지만, 일관된 결과를 얻기 위해서 최소 10개 이상의 문장은 평가해봐야 한다.
- 실험할 때 마다 맥락은 비슷하지만 결과값이 조금씩 변한다.
- 문서의 구조나 내용마다 성능이 다를 수 있다.