****
# *Machine Learning*
#### [Entropy, KL Divergence](https://github.com/hchoi256/ai-terms/blob/main/entropy.md)

#### [시계열 분석](https://github.com/hchoi256/ai-terms/blob/main/time-series-analysis.md)
#### [Overfitting](https://github.com/hchoi256/ai-terms/blob/main/overfitting.md)
#### 차원의 저주
<details markdown="1">
<summary></summary>
차원이 증가하면서 학습데이터 수가 차원 수보다 적어져서 성능이 저하되는 현상이다.
</details>

#### 표준화 vs. 정규화 vs. MinMaxScaler
<details markdown="1">
<summary></summary>
**표준화**: 평균이 0이고 분산이 1인 가우시안 정규분포를 가진 값으로 변환한다.

**정규화**: 모두 동일한 크기 단위로 비교하기 위해 값을 모두 0과 1사이(음수가 있을경우 -1과 1사이)의 값으로 변환한다.

**MinMaxScaler**: 데이터가 정규분포를 따르지 않아도 될 때, 0과 1사이의 범위값으로 변환한다.
</details>

#### k-fold 교차 검증
<details markdown="1">
<summary></summary>
테스트셋으로 모델 성능을 평가하는 것에 치중되어서 테스트셋 과적합이 발생할 수 있다 (한 테스트셋에 대해서만 성능이 좋게 나왔을수도). 따라서, 훈련 데이터셋에 대하여 일정 비율을 fold로 분류하여 테스트 검증과 별개로 따로 교차 검증을 수행한다.
</details>

#### 전이학습(Transfer Learning)
<details markdown="1">
<summary></summary>
전이 학습(Transfer Learning)은 특정 분야에서 학습된 신경망의 일부 능력을 유사하거나 전혀 새로운 분야에서 사용되는 신경망의 학습에 이용하는 것을 의미한다.
</details>

#### 강화학습
<details markdown="1">
<summary></summary>
주어진 상황에서 어떤 행동을 취할지 **보상** 심리를 기반으로 하는 Greedy algorithm으로 학습한다.
</details>

#### GAN
<details markdown="1">
<summary></summary>
비지도학습에 사용되는 머신러닝 프레임워크의 한 종류로, 생성자와 구분자가 서로 대립하며(Adversarial:대립하는) 서로의 성능을 점차 개선해 나가는 쪽으로 학습이 진행되는 것이 주요 개념이다.
</details>

****
# *Deep Learning*
#### [Convolution](https://github.com/hchoi256/ai-boot-camp/blob/main/ai/deep-learning/cnn.md)
<details markdown="1">
<summary></summary>
이미지에서 feature를 뽑기위해 사용하는 합성곱 연산 과정이다.
</details>

#### Softmax vs. Sigmoid (분류)
<details markdown="1">
<summary></summary>
Softmax - 다중분류

Sigmoid - 이진분류
</details>


#### [Gradient Descent and Stocastic Gradient Descent](https://github.com/hchoi256/lg-ai-auto-driving-radar-sensor/blob/main/supervised-learning/gradient-discent.md)

#### [Ensemble](https://github.com/hchoi256/lg-ai-auto-driving-radar-sensor/blob/main/supervised-learning/ensemble.md)

#### Fully Connected Layer(= hidden layer)
<details markdown="1">
<summary></summary>
발견한 특징점을 기반으로 이미지를 분류하는 레이어 구간
</details>

#### Batch and Epoch
<details markdown="1">
<summary></summary>

![image](https://user-images.githubusercontent.com/39285147/179927500-1f89d8b9-f1d0-409d-b935-d54028e00113.png)

**Batch**: 전체 트레이닝 데이터 셋을 여러 작은 그룹으로 나누었을 때, 하나의 소그룹에 속하는 데이터 수를 의미한다.
- Batch 사이즈 ↑, 한 번에 처리해야할 양 ↑, 학습 속도가 ↓, 메모리 부족
- Batch 사이즈 ↓, 적은 샘플을 참조해서 가중치 업데이트가 빈번하게 일어나기 때문에, 비교적 *불안정하게* 훈련될 수도 있습니다.

**Epoch**: 딥러닝에서는 epoch은 전체 트레이닝 셋이 신경망을 통과한 횟수이다. 가령, 1-epoch는 전체 트레인이 셋이 하나의 신경망에 적용되어 순전파와 역전파를 통해 신경망을 한 번 통과했다는 것을 의미한다.
</details>

#### Padding, Stride, Pooling
<details markdown="1">
Padding: ( n - f + 1 ) x ( n - f + 1 )
- edge 부분 픽셀은 한 번만 사용되어 윤곽 정보 소실을 방지한다.
- 이미지 축소를 방지한다

Stride
![image](https://user-images.githubusercontent.com/39285147/179364041-af7c8918-1e2b-49a2-b8d1-147b1b3aff8b.png)

- 필터 적용시 이동 간격

Pooling
- 사이즈감소 및 노이즈 제거

> convolution layer의 경우 원본이미지 크기를 유지하면서 depth를 키우기 때문에 메모리를 많이 차지한다. 따라서, 특징은 유지하면서 데이터와 feature map의 사이즈를 줄임으로써 용량을 절약할 수 있다. 

<summary></summary>

</details>

#### Fine-tuning
<details markdown="1">
<summary></summary>
기존에 학습되어진 모델을 기반으로 아키텍쳐를 새로운 목적에 맞게 변형하고 이미 학습된 모델 Weights로 부터 학습을 업데이트 하는 방법이다.
</details>

#### Zero-shot Learning
<details markdown="1">
<summary></summary>
Zero shot은 train set에 포함되지 않은 unseen class를 예측하는 분야로, test시에 unseen data를 입력 받아도, seen data로 학습된 지식을 전이하여 unseen data를 unseen class로 예측할 수 있다.
- **CV**: 클래스 레이블 간의 표현 유사성에 의존

- **NLP**: 동일한 의미적 공간에서의 레이블을 나타내는 '라벨 이해' 기능 기반

</details>

#### Black box
<details markdown="1">
<summary></summary>
Black box란 결과는 인간과 유사하게 또는 원하는대로 도출할 수 있지만 어떻게, 무엇을 근거로 그러한 결과가 나왔는지 알 수 없는 것
</details>

#### GAP
<details markdown="1">

![image](https://user-images.githubusercontent.com/39285147/179212153-3e0a7ee0-a531-43ff-a0e4-64254f9aa362.png)

GAP layer는 각각의 feature map의 값들을 평균을 취한 것으로, feature map의 크기와 관계없이 channel이 k개라면 k개의 평균 값을 얻을 수 있다.
- GAP는 FC Layer와 달리 연산이 필요한 파라미터 수를 크게 줄일 수 있다
  - Regulariztion과 유사한 동작을 통해 overfitting을 방지할 수 있다.
- FC layer는 Convolution layer에서 유지하던 위치정보가 손실되는 반면, GAP layer는 위치정보를 담고 있기 때문에 localization에 유리하다.
<summary></summary>
</details>

#### Grad_CAM
<details markdown="1">

![image](https://user-images.githubusercontent.com/39285147/179212552-2f937eab-1a74-4fa9-9a98-5e27c0db0f39.png)

어떤 target concept일지라도 final convolutional layer로 흐르는 gradient를 사용하여 이미지의 중요한 영역을 강조하는 localization map을 만든다.
<summary></summary>
</details>


#### Back Propagation
<details markdown="1">
역전파 알고리즘은 출력값에 대한 입력값의 기울기(미분값)을 출력층 layer에서부터 계산하여 거꾸로 전파시키는 것이다.

전 layer들에서의 기울기와 서로 곱하는 형식으로 나아가면서 최종적으로 출력층에서의 output값에 대한 입력층에서의 input data의 기울기 값을 구할 수 있다.

1. 이렇게 기울기(미분값)을 구하는 이유?
- 역전파 알고리즘으로 각 layer에서 기울기 값을 구하고 그 기울기 값을 이용하여 Gradient descent 방법으로 가중치 w와 b를 update시키면서 파라미터가 매우 많고 layer가 여러개 있을때 학습하기 어려운 문제를 해결한다.

2. 최종 output에 대한 기울기 값을 각 layer별로 구하는 이유?
- 각 layer의 node(parameter)별로 학습을 해서 각 가중치(기울기 값)을 업데이트한다.

<summary></summary>
</details>

****
# *Statistic / Probability*

#### Central Limit Theorem
<details markdown="1">
<summary></summary>

##### 정의
모집단(평균: μ, 표준편차: σ)이 어떤 분포를 따르던지 무관하게, 표본평균의 표본분포는 n이 커지면(>= 30) 평균이 μ이고 표준편차가 σ/n인 **정규분포**를 따른다.

##### 의의
표본평균을 통해서 모집단의 모수인 모평균과 모표준편차를 추정할 수 있는 확률적 근거 제시
</details>

#### Law of Large Numbers (LLN)
<details markdown="1">
<summary></summary>
경험적 확률과 수학적 확률 사이의 관계를 나타내는 법칙; 표본집단의 크기가 커지면 그 표본평균이 모평균에 가까워짐을 의미
</details>

#### Normalization vs. Standardization
<details markdown="1">
<summary></summary>
정규화: [0, 1] 분포

표준화: 평균 0, 표준편차 1 분포
</details>

#### d-separation (방향성 독립)
<details markdown="1">
<summary></summary>
방향성 그래프 모형에서 어떤 두 노드(확률변수)가 조건부 독립인지 아닌지 알아보는 방법
</details>

#### 조건부 독립
<details markdown="1">
<summary></summary>
P(A,B|C) = P(A|C)*P(B|C) , (AㅛB)|C 으로 표기되며, 조건부 독립. A와 B 사건은, C사건 하에서는 서로 독립이다
</details>

#### [Bias-Variance Trade-off](https://github.com/hchoi256/lg-ai-auto-driving-radar-sensor/blob/main/supervised-learning/sl-foundation.md)

#### 계층적샘플링
<details markdown="1">
<summary></summary>
모집단의 데이터 분포 비율을 유지하면서 데이터를 샘플링(취득)하는 것을 말합니다
</details>

****
# *Lienar Algebra*

