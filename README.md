# Book Rating Prediction (RecSys)

본 프로젝트는 Upstage AI Stage에서 주최한 대회 기반으로 수행하였습니다.
사용자의 과거 책 평점 데이터를 기반으로 사용자-책(user-item) 쌍에 대한 평점을 예측하는 추천 시스템입니다.



## Problem Overview
- 추천 시스템(RecSys) 문제
- 사용자에게 아직 평가되지 않은 책에 대한 예상 평점을 예측
- Explicit Feedback(1~10점)을 사용하는 Regression 문제

## Evaluation
- 평가 지표: RMSE
- 예측한 평점과 실제 평점 간의 오차를 기반으로 평가
- RMSE 값이 낮을수록 모델 성능이 우수

## Dataset
본 데이터는 대회에서 제공된 공개 데이터셋을 사용하였습니다.
- 사용자-책 평점 데이터 ('train_ratings.csv')
- 사용자 메타데이터 ('users.csv')
- 책 메타데이터 ('books.csv')
- 책 표지 이미지 데이터
