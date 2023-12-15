# RAG
- 기본 RAG와 Sentece-window, Auto-merging 기법을 활용한 advanced RAG 기법에 대해서 [Trulens](https://github.com/truera/trulens)를 이용하여 RAG 성능을 비교하는 프로젝트
## Basic RAG
- 사용자에게 보다 정확한 Response를 제공하기 위해 추가적인 private 또는 real-time data 같은 추가적인 context를 제공하고, 이를 기반으로 response를 생성하는 방법이다.
- 보통 문서를 불러와서 split하여 chunk를 생성, 이를 임베딩하여 vector database에 저장한 후 retrieve하여 사용자 프롬프트(쿼리)와 유사한 context를 제공하여 응답을 생성하는 구조이다.

<table><tr><td><img src='https://python.langchain.com/assets/images/rag_indexing-8160f90a90a33253d0154659cf7d453f.png' width=4200></td><td><img src='https://python.langchain.com/assets/images/rag_retrieval_generation-1046a4668d6bb08786ef73c56d4f228a.png' width=4200></td></tr></table>  

- 하지만 retrieve할 때 chunk로 쪼개진 정보를 그대로 사용하여 충분한 context를 제공하지 못한다는 문제점이 있다.

# RAG Triad
![RAG Triad](https://github.com/jjlee6496/RAG/blob/main/imgs/readme/RAG_Triad.png)
- RAG 성능을 평가하기 위해 다음 3가지 metric을 사용한다. 이를 RAG Triad라고 부른다.
## [Anaswer Relevance](https://github.com/jjlee6496/RAG/blob/main/utils.py#L135)
- Output Response가 user query와 얼마나 relevant한지 CoT로 판단한다.
## [Context Relevance](https://github.com/jjlee6496/RAG/blob/main/utils.py#L141)
- Output Response가 context(retrieved chunks)에 얼마나 relevant한지 CoT로 판단한다.
## [Groundedness](https://github.com/jjlee6496/RAG/blob/main/utils.py#L152)
- Output response가 context(retrieved chunks)에 얼마나 기반하는지에 대해 평가.
- Retrieval step에서 충분히 많은 관련된 context를 찾지 못한다면 context에 드러난 정보들을 기반으로 답변하는 것이 아닌 pre-trained information을 사용하게 된다. 이는 Groundedness를 떨어뜨린다.

# Sentence-window
## Result
- Sentence-window 기법의 window 크기를 1, 3, 5로 늘려가면서 RAG Triad에 대한 성능 평가
![실험결과1](https://github.com/jjlee6496/RAG/blob/main/imgs/test/sentence_window_comparison.png)
- window size가 커질수록 
# Auto-merging

## Result

# Reference
- https://python.langchain.com/docs/use_cases/question_answering/
- https://learn.deeplearning.ai/building-evaluating-advanced-rag/lesson/1/lesson_1

# To-do
- [X] RAG 개념 정리
- [ ] Basic RAG 코드 작성
- [ ] RAG Triad 개념 정리
- [X] Sentence-window Retrieval 개념 정리
- [X] Sentence-window Retrieval 코드 작성
- [ ] Auto-merging Retrieval 개념 정리
- [ ] Auto-merging Retrieval 코드 작성
- [X] Evaluation 코드 작성
- [ ] 전체 비교 실험 진행
- [ ] 한국어 문서 비교 실험 진행
