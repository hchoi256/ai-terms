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

*표준화*: 평균이 0이고 분산이 1인 가우시안 정규분포를 가진 값으로 변환한다.

*정규화*: 모두 동일한 크기 단위로 비교하기 위해 값을 모두 0과 1사이(음수가 있을경우 -1과 1사이)의 값으로 변환한다.

*MinMaxScaler*: 데이터가 정규분포를 따르지 않아도 될 때, 0과 1사이의 범위값으로 변환한다.
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

#### 신경망 학습에서 '정확도'가 아닌 '손실함수' 사용 이유?
<details markdown="1">
<summary></summary>
최적의 매개변수(가중치와 편향)을 탐색할 때 손실함수에서는 미분을 통하여 손실함수의 값을 작게하는 매개변수를 탐색하지만, 정확도를 지표로 삼는 경우 그 미분값이 대부분의 장소에서 0이 되므로 매개변수 갱신이 어렵다.

손실함수의 예로는 **평균제곱오차(회귀), 크로스 엔트로피(분류)** 등이 있다.
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

일반적으로 딥러닝은 training에 사용된 class만을 예측할 수 있다. 따라서 unseen data가 입력되면 seen class로 예측하는 바보가 되버리는데, Zero shot은 train set에 포함되지 않은 unseen class를 예측하는 분야이다.

즉, unseen data를 입력 받아도, seen data로 학습된 지식을 전이하여 unseen data를 unseen class로 예측할 수 있다.

- **CV**: 클래스 레이블 간의 표현 유사성에 의존

- **NLP**: 동일한 의미적 공간에서의 레이블을 나타내는 '라벨 이해' 기능 기반

</details>

#### Black box
<details markdown="1">
<summary></summary>
Black box란 결과는 인간과 유사하게 또는 원하는대로 도출할 수 있지만 어떻게, 무엇을 근거로 그러한 결과가 나왔는지 알 수 없는 것
</details>

#### GAP vs GMP
<details markdown="1">

![image](https://user-images.githubusercontent.com/39285147/180636329-dc5f6258-ced6-47c0-ad0b-e76c28804db2.png)

- Global Max Pooling은 탐지 사물을 포인트로 짚는 반면, GAP는 사물의 위치를 범위로 잡아내는 장점이 있다.
  - **Average pooling method**: smooths out the image and hence the sharp features may not be identified when this pooling method is used.
  - **Max pooling**: brighter pixels from the image.

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

#### 확률 vs. 통계
<details markdown="1">
<summary></summary>

*확률*: **하나의** 사건이 발생할 경우

*통계*: **여러 개의** 사건이 발생할 확률

</details>

#### pmf vs. pdf vs. cdf
<details markdown="1">
<summary></summary>

*pmf*(확률질량함수): 어떤 사건이 발생할 이산형 확률분포이다.

*pdf*(확률밀도함수): 어떤 사건이 발생할 연속형 확률분포이다.

*cdf*(누적분포함수): pdf 확률값들이 누적된 확률분포이다.

> pdf를 적분하면 cdf가 되고, cdf를 미분하면 pdf가 된다 (cdf에서 어느 지점까지의 넓이가 pdf에서 그 지점의 확률이다)..

</details>

#### 이산확률분포 vs. 연속확률분포
<details markdown="1">
<summary></summary>

![image](https://user-images.githubusercontent.com/39285147/180481741-fee61485-7d1c-42f8-825d-3ecec28427bc.png)

### 이산확률분포 (Binomial, Bernoulli, Multinomial, Multinoulli, Geometric, Poisson, Hypergeometric)
pmf를 통해 표현 가능하며, 확률 변수가 가질 수 있는 값이 가산 개 있다.

*Binomial*(이항분포): 독립된 연속된 N번의 베르누이 시행에서 시행 확률(p)를 반복했을 때의 이산확률분포이다 (i.e., 동전던지기)

*Negative-Binomial*(음이항분포):  음이항분포는 기하분포를 일반화한 버전으로, 기하분포는 처음 성공까지를 보지만, 음이항분포는 r번째 성공까지이다.

*Bernoulli*(베르누이): 이항분포에서 시행 횟수(N)이 1일 때이다.

*Multinomial*(다항분포): 베르누이 시행은 발생 가능한 경우의 수가 두 가지였지만, 다항분포는 발생 경우의 수가 그 이상인 경우를 말한다 (i.e., 주사위).

*Multinoulli*: 다항분포에서 시행 횟수(N)이 1일 때이다.

*Geometric*(기하분포): 어떤 행위를 처음 성공할때까지 시도하는데, 처음 성공할때까지의 시도횟수 또는 실패한 횟수의 분포이다.

*Poisson*(포아송분포): 률론에서 **단위 시간 안에** 어떤 사건이 몇 번 발생할 것인지를 표현하는 이산확률분포이다.

*Hypergeometric*(초기하분포): 비복원추출에서 N개 중에 M개가 원하는 것이고, K번 추출했을때 원하는 것 x개가 뽑힐 확률의 분포이다 (i.e., 흰/검 공이 들어있는 항아리에서 흰 공을 k개 뽑을 확률분포).

### 연속확률분포 (정규분포, 감마분포, 지수분포, 카이제곱분포, 베타분포, 균일분포)
pdf를 통해 표현 가능하며, 확률 변수가 가질 수 있는 값이 셀 수 없다.

*Normal*(정규분포): 평균과 표준편차를 기준으로 종모양 분포를 나타낸다. 여기서, 평균이 0이고 표준편차가 1인 분포를 표준정규분포라 지칭한다.

*Gamma*(감마분포): α 번째 사건이 발생할 때 까지의 대기시간의 분포

*Exponential*(지수분포): 첫번째 사건이 발생할 때 까지의 대기시간의 분포

*Chi-squared*(카이제곱분포): 감마분포의 특수한 경우로 (α=p/2,β=2), 두 범주형 변수에 대한 분석 방법이다. 카이제곱분포는 분산의 특징을 확률분포로 만든 것이므로 집단의 분산을 추정하고 검정할 때 많이 사용된다. 보통 0에서 시작되는 positively skewed 형태의 분포모양을 띄는데, 이는 0에서 멀어질수록 분산의 크기가 큰 경우(가령 키 차이가 50cm 이상 나는 경우)가 적다는 의미이다. 

*Betea*(베타분포): 두 매개변수 α 와 β 에 따라 [0, 1] 구간에서 정의 되는 **단일** 확률변수에 대한 연속확률분포이다.

*Dirichlet*(디리클레분포): 두 매개변수 α 와 β 에 따라 [0, 1] 구간에서 정의 되는 **다변수** 확률변수에 대한 연속확률분포이다.

*Uniform*(균일분포): 특정 범위 내에서 균등하게 나타나 있을 경우를 가리킨다.

</details>


#### 베이지안 이론이란?
<details markdown="1">
<summary></summary>
이미 사건이 일어났고(i.e., 창고의 불량 청바지), 사건발생의 원인에 대한 확률(i.e., 사후확률 = 이 창고의 불량 청바지는 어떤 공장에서 불량생산되어 온것일까?)을 사건발생전에 이미 알고 있는 정보(i.e., 사전확률 = 구미, 청주, 대구 공장의 불량률)을 이용하여 구하는 것이라 하겠다. 
</details>

#### Unbiased Estimation의 장점은 무엇이며, 무조건 좋은건지?
<details markdown="1">
<summary></summary>

불편추정치(unbiased estimation)은 편차(추정값들의 기대치와 실제 기대치와의 차이)가 0인 경우를 일컫는다. 이를 통해 우리는 모평균을 **정확하게** 도출해낼 수 있다.

그럼 무조건적으로 불편추정치라는 것이 좋은 의미일까? 답은 아니다. 여기서 주목해야할 부분은 추정값들의 평균과 실제값의 차이이다. unbiased estimation이라 할지라도 추정값들의 분산은 매우 클수도 있으므로 절대적으로 신뢰할 수는 없다. 

> *Bias-Variance Tradeoff* 정리를 참고하면 이해가 더 수월할 것이다.

</details>

#### 주변확률분포(Marginal Distribution)과 조건부 분포(Conditional Distribution)
<details markdown="1">
<summary></summary>
*주변확률분포*: **하나의** 확률변수에 대한 결합확률들을 모두 합한다.

*조건부 분포*: 어떤 사건 B가 일어났을 때 사건 A가 발생할 확률이다; P(B|A)

![image](https://user-images.githubusercontent.com/39285147/180476247-0371081a-8563-4ff8-9f83-e0a70a472676.png)

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

#### Sample Variance를 구할 때, N대신에 N-1로 나눠주는 이유는 무엇인가?
<details markdown="1">
<summary></summary>

**1. 모평균과의 정확도 근사를 위해서**

표본을 무한정 추출하면 표본분산과 표본평균은 모분산과 모평균에 수렴하여야 하지만, 실제로 N으로 나누어 표본분산을 구할 경우 표본분산보다 모분산이 더 큰 현상이 발생한다. 따라서, 표본분산과 모분산의 차이를 줄이기 위하여 표본분산의 크기를 키우고자 분모에 N 대신 N-1을 분모에서 사용한다.

**2. 자유도 (Degree of Freedom)**

분산은 편차 제곱의 평균이므로, 표본평균을 알고 있다는 전제로 도출할 수 있는 값이다. 따라서, 편차 제곱의 평균을 구할 때 분모에 N 대신 N-1을 사용하면, 우리는 표본평균을 알고있기 때문에 마지막 추정값을 더 나은 통계치 도출을 위해 자유롭게 제외할 수 있다. 이를 토대로, 우리는 표본분산의 자유도가 N-1임을 알 수 있다.

</details>

#### 상관관계와 공분산
<details markdown="1">
<summary></summary>

*공분산*: 두 개의 확률변수의 상관정도를 나타내는 값이다 [-1, 1]. 공분산의 크기는 두 확률변수의 scale에 크게 영향을 받는다.

*상관관계*: 두 변수 간에 선형 관계의 정도를 수량화하는 측도이다. 이때 두 변수간의 관계의 강도를 상관계수(correlation coefficient)라고 한다. 만약, 상관계수가 0이면 두 확률변수는 아무런 선형 상관관계를 갖지 않는다.

공분산 vs. 상관관계
- 공분산: 상관 정도의 절대적인 크기를 측정 X
- 상관관계: 상관 정도의 절대적인 크기를 측정 O

</details>

#### [MLE(최대우도법)](https://github.com/hchoi256/ai-terms/blob/main/mle.md)


####  Conjugate Prior
<details markdown="1">
<summary></summary>

사후확률이 사전확률과 동일한 함수형태를 가지도록 해준다.

베이즈 확률론에서 사후확률을 계산함에 있어 사후 확률이 사전 확률 분포와 같은 분포 계열에 속하는 경우 그 사전확률분포는 켤레 사전분포(Conjugate Prior)이다.

</details>

#### False Positive
<details markdown="1">
<summary></summary>

![image](https://user-images.githubusercontent.com/39285147/180516436-f1ecd0b6-1e24-461c-8261-8946045a22ff.png)

</details>

****
# *Lienar Algebra*

