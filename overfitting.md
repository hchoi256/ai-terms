# Overfitting
![image](https://user-images.githubusercontent.com/39285147/178137619-ecba42bd-5e89-4fbd-af2c-d408f40a7be9.png)

훈련 데이터에 대한 정확도는 높을지라도, 새로운 데이터(Test Set)에 대해서는 제대로 동작하지 않음

# Solution
## 1. 데이터 양 늘리기
데이터를 증가시켜 데이터 복잡도를 모델 복잡도보다 우위로 만든다.

데이터 양이 적을 경우 의도적으로 기존 데이터를 변형한다
- *Data Augmentation*
- 이미지 회전, 노이즈 등
- 텍스트 데이터: 역번역(*Back Translation*) 등

## 2. 모델 복잡도 줄이기
인공신경망의 복잡도는 은닉층의 및 매개변수의 수로 결정되므로, 이것들을 줄인다.

## 3. 가중치 규제(Regularization)
복잡한 모델일수록 매개변수의 계수의 절대값이 크다. 따라서, 이 값들을 가중치 규제를 통해 0에 수렴하게 만듦으로써 모델에 주는 영향을 제거한다.
- L1 Regularization ![image](https://user-images.githubusercontent.com/39285147/178137782-97d720b1-f286-4f78-939c-62630066d3d6.png)
  - 가중치 0으로 만듬
- L1 Regularization ![image](https://user-images.githubusercontent.com/39285147/178137788-8e1849fb-75bf-4967-9a88-32a0e2feea52.png)
  - 가중치 0에 가깝게 만듬

## 4. Drop-out
![image](https://user-images.githubusercontent.com/39285147/178137809-e0bce3e4-a1ff-413c-85f6-30bc7d29e598.png)

신경망 일부를 사용하지 않는 방법

## Underfitting
학습 데이터에 대한 정확도가 낮음

# Reference
[here](https://wikidocs.net/61374)
