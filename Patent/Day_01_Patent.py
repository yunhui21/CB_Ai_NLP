# Day_01_Patent.py
'''
Research
* patent.google.com
구글에서 3년전부터 특허관련 자료검색에 머신러닝 적용.
resources patent.google.com : set관리하는 팀. 접근성확장에 대한 고민.
BigQuery PUblic Datsset에 특허 데이터 오픈
Re_KX_bmR0(youtube): 빅데이터 set 공개.
https://media.epo.org/play/gsgoogle2017
* Scholarly Big Data
- 특허가 아닌 일바넉으로 논문 데이터를 다루는 AI분야
- 논문 검색, 유사 논문 군집, Citation 분석, 지표화 등.
- 특허와 논문은 기본 데이터의 유형이 비슷하여 연구 주제가 공유 가능
- 특허에 비해 데이터가 비정형이거나 확보의 어려움이 있음.
- 특허에 비해 참여하는 연구자가 많음.
: AI2 Sematic Scholar 연구를 많이 공개.
Allen AI

* Deep learning for Patent Analysis
- 전통적인 Text-mining 문제인 문서 분류 및 군집 문제
: 기존 텍스트 데이터와 달리 특허 특유의 데이터 타입을 활용
- 정보 시각화 이슈 - Patent Map(or Patent Landscaping)
- 법률 이슈, Claim, Citation, Patent Infraigement, Generation
- 이미지 처러: Figure Generation, Figure Searching.
- 특허 분석은 AI 분야중 Law AI에 포함되어 있음.
- 구글 외에도 몇 개의 스타트업과 AON 같은 보험사에도 팀을 구축함.
- 트렌드나 Future Analysis 보다는 답이 있는 정형화된 문제에  초점.

* dataSet
(why dataset is important?)
- 특허 AI를 구축을 위해서는 데이터 셋 확보가 중요
- 영역별로 표준호된 Benchmarking dataset이 필요함.
- Imagenet, SQuAD등 표준화된 데이터 셋은 알고리즘 발전의 밑거름.
- 특허 영역의 세부 Task를 구축 필요
(From Trend Analysis to Comparable AI Task)
- Trend Analysis        Comparable AI Task
- 중요한 연구 이슈         - 알고리즘 상 성능 비교가 가능한 특허 분석 TASK
- 기업별 동향            - 특허 분류, 검색, 데이터 정제
- 미래 연구 방향 도출       - Benchmarking Dataset 필요
(Task& Dataset)
- patent embeddings.
- passage reitrieval statring from claims
- Structure Recognition
- Prior Art Candidate Search Task
- Assignee Disambiguation
- Patent graph
(Raw Dataset)
- Task에 따라서는 연구자가 직접 데이터 셋을 구축해야함.
-> 최근 NLP 연구의 공통적인 추세
- 연구 문제 정의 -> 데이터셋 구축
- 데이터 셋 구축 도구
1) 직접하기(google bulk patent dataset, reedtech dataset)
XML-based raw patent documents
2) Google Big Query - SQL - based ETL tool
readtech : xml자료이므로 전처리가 오래 걸림.
- human - labeled dataset
- 필요에 따라 파일화 되어 있지 않은 태깅 데이터 사용 가능
- 기존 회사내 특허 분류 데이터, 법원내 특허 침허 건 등 활용 가능
- Big query 내 Litigation dataset 활용 가능.
* Today
'''