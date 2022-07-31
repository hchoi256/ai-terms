# 시계열
## 시계열데이터
일정 시간 간격으로 배열되어 있는 데이터
- 규칙적 패턴
- 불규칙한 패턴


## 구성요소
![image](https://user-images.githubusercontent.com/39285147/178138244-ed2e2426-95ca-41de-85fb-ae53cfba8edf.png)

## 시계열 분석
실제 세상에는 시간적 요소가 중요한 데이터들이 많아서, 시계열 분석은 ML에서 중요한 영역으로 다뤄진다.
- 시간 변수 1개 + 여러 개의 변수

## 시계열 데이터의 ML 모델
### 목표
- 잔차제곱합과 같은 손실 함수(loss function)를 최소화 함으로써 예측 정확도를 향상시키는 것

### 관련 연구
- 최근에는 딥러닝을 이용하여 시계열 데이터의 연속성을 찾아내는 방법이 연구되고 있다; RNN 종류의 LSTM이 좋은 성능을 나타낸다.

## ML/DL Algorithms w/ 시계열
- **머신러닝**
  - Multi-Layer Perceptron (MLP)
  - Bayesian Neural Network (BNN)
  - Radial Basis Functions (RBF)
  - Generalized Regression Neural Networks (GRNN)
  - kernel regression K-Nearest Neighbor regression (KNN)
  - CART regression trees (CART)
  - Support Vector Regression (SVR)
  - Gaussian Processes (GP)
- **딥러닝**
  - Recurrent Neural Network (RNN)
  - Long Short-Term Memory (LSTM)
    - 미세먼지 농도 예측 알고리즘, etc.
  - [ARIMA](https://byeongkijeong.github.io/ARIMA-with-Python/)

# Reference
https://min23th.tistory.com/4?category=954545
