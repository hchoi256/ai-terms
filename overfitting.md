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
![image](https://user-images.githubusercontent.com/39285147/184464391-62711156-ddf3-4413-96cd-0c4c884c16e5.png)

가중치 규제를 통해 0에 수렴하게 만듦으로써 해당 뉴런이 모델에 주는 영향을 제거한다.

### L1 규제 (Lasso)
규제가 커질 수록 훈련 세트의 손실과 검증 세트의 손실이 커지고 (= underfitting), 규제가 커질 수록 가중치 값이 "0"에 가까워진다. 

### L2 규제 (Ridge)
L1 규제와 비슷한 양상을 보이나, 규제가 강해져도 과소 접합이 그렇게 심해지지 않는 특성을 가지고 있다. 그래서, **L2 규제를 많이 사용한다.**

## 4. Drop-out
![image](https://user-images.githubusercontent.com/39285147/178137809-e0bce3e4-a1ff-413c-85f6-30bc7d29e598.png)

신경망 일부를 사용하지 않는 방법

## 5. Ensemble
average over a number of models

## Underfitting
학습 데이터에 대한 정확도가 낮음

# Reference
[here](https://wikidocs.net/61374)
