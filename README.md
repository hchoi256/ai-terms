****
# *Machine Learning*
#### [Entropy, KL Divergence](https://github.com/hchoi256/ai-terms/blob/main/entropy.md)

#### [시계열 분석](https://github.com/hchoi256/ai-terms/blob/main/time-series-analysis.md)
#### [Overfitting & How to Avoid Overfitting](https://github.com/hchoi256/ai-terms/blob/main/overfitting.md)
#### 차원의 저주
<details markdown="1">
<summary></summary>
차원이 증가하면서 학습데이터 수가 차원 수보다 적어져서 성능이 저하되는 현상이다.
</details>

#### Activation Function의 종류 세 가지
<details markdown="1">
<summary></summary>

활성화함수란 입력 신호의 총합을 출력 신호로 변환하는 함수로, 입력 받은 신호를 얼마나 출력할지 결정하고 Network에 층을 쌓아 비선형성을 표현할 수 있다. 

### 1. Ridge activation Function
Multivariate functions acting on a linear combination of the input variable
- Linear
- ReLU
- Logistic

### 2. Radial activation Function(원형기준함수)

![image](https://user-images.githubusercontent.com/39285147/182003709-d319e437-6250-4edd-97e7-c473864edc86.png)

평균과 분산을 가지는 정규분포의 데이터의 분포를 근사한다. kernel function을 사용하여 비선형적인 데이터 분포에 대한 처리가 가능하며, MLP보다 학습이 빠르다
- Gaussian

### 3. Folding activation Function
Folding activation functions are extensively used in the pooling layers in convolutional neural networks, and in output layers of multiclass classification networks. These activations perform aggregation over the inputs, such as taking the mean, minimum or maximum. In multiclass classification the softmax activation is often used.
- Softmax

</details>

#### Netwon's method
<details markdown="1">
<summary></summary>

값을 계속 대입하여 함수값을 0(f(x) = 0인 x)으로 만들어주는 값인 해를 구하는 방법 중 하나이다.

현재 x값에서 접선을 그리고 접선이 x축과 만나는 지점으로 x를 이동시켜 가면서 점진적으로 해를 찾는 방법이다.

2차 방정식의 인수분해와 비슷해보이지만, 7차 방정식의 경우 인수분해가 어려워, 뉴턴법을 사용한다.


</details>

#### Decision Theory (의사결정)
<details markdown="1">
<summary></summary>

불확실성에 직면하여 결정을 내리지 않으면 안 될 경우, 어떤 결정을 해야할 것이며, 또 어떤 정보를 어떻게 이용해야 하는가에 관한 문제에 답하려는 통계적 결정이론이다.

기대효용이 최대가 되도록 결정하는 것을 '연역'이라 일컫는다. 결정자에게 불확실성 하에서 합리적이고 가장 적절한 결정을 도출하는 것이다.

#### Applications
Naive Beyas 정리를 토대로 사전확률을 통한 사후확률 도출이라는 구체화 과정을 거친다.

</details>

#### ROC (Receiver Operating Characteristic)
<details markdown="1">
<summary></summary>

![image](https://user-images.githubusercontent.com/39285147/180647724-6bd69d98-6ae4-46f5-b28c-255d5acc95dd.png)

ROC 곡선은 Binary Classifier System에 대한 성능 평가 기법으로, 모델이 맞게 예측한 TP를 y축으로 틀리게 예측한 FP를 x축으로 하여 도표를 그린다.

</details>

#### Semi-Supervised Learning
<details markdown="1">
<summary></summary>

준지도학습은 소량의 labeled data에는ㄴ supervised learning을 활용하고, 소량의 unlabeled data 혹은 대용량 unalbeled data에 대하여 unsupervised learning을 적용해 추가적인 성능 향상을 목표로 한다.

기존 지도학습의 label 종속성에서 다소 벗어나 '데이터 자체의 본질적인 특성'을 모델링하여 소량의 labeled data를 통한 약간의 가이드로 일반화 성능을 끌어올린다.

#### Assumptions
1. smoothness: 같은 class/cluster에 위치한 두 입력이 입력공간 상에서 고밀도 지역에 위치한다면, 해당 출력도 가까울 것이다.
2. cluster: 데이터 포인트들이 같은 cluster에 있다면, 그들은 같은 class일 것이다.
3. manifold: 고차원 데이터를 저차원 manifold로 보낼 수 있다 (고차원에서는 거리가 비슷하여 분류가 어려워서 저차원으로 맵핑해야 한다).

#### Applications
- CIFA-100
- ImageNet

</details>


#### 표준화 vs. 정규화 vs. MinMaxScaler
<details markdown="1">
<summary></summary>

*표준화*: 평균이 0이고 분산이 1인 가우시안 정규분포를 가진 값으로 변환한다.

*정규화*: 모두 동일한 크기 단위로 비교하기 위해 값을 모두 0과 1사이(음수가 있을경우 -1과 1사이)의 값으로 변환한다.

*MinMaxScaler*: 데이터가 정규분포를 따르지 않아도 될 때, 0과 1사이의 범위값으로 변환한다.
</details>

#### k-fold 교차 검증 (cross validation)
<details markdown="1">
<summary></summary>
하나의 테스트셋으로 모델 성능을 평가하는 것에 치중된 테스트셋 과적합을 방지할 수 있다 (한 테스트셋에 대해서만 성능이 좋게 나왔을수도). 

따라서, 훈련 데이터셋에 대하여 일정 비율을 fold로 분류하여 테스트 검증과 별개로 하나씩 테스트셋으로 사용하면서 교차 검증을 수행한다.

이를 통해, 데이터의 100%를 테스트셋으로 활용하면서 총 K개의 성능 결과를 내고, 이 K 값들의 평균을 해당 모델의 성능으로 배출한다.

</details>

#### 전이학습(Transfer Learning)
<details markdown="1">
<summary></summary>

전이 학습(Transfer Learning)은 특정 분야에서 학습된 신경망의 일부 능력을 유사하거나 전혀 새로운 분야에서 사용되는 신경망의 학습에 이용하는 것을 의미한다.

전이 학습을 통해서 더 많은 지식을 얻음으로써 성능도 향상시키고 학습 속도도 빠르게 만들 수 있다.

성능을 향상시키려면 기본적으로 데이터가 많아야 하는데, 데이터 부족을 극복하기 위해 다른 지식 데이터를 가져온다.
- 학습된 다른 데이터를 가져올 수도 있고, 학습된 다른 모델을 가져올 수도 있다.

### 전이 학습 장점
- 보다 적은 데이터 양으로 성능 개선 가능
  - 적은 우리 데이터 + 많은 소스 데이터
- 학습 시간 절약

[reference](https://minimi22.tistory.com/31)

</details>

#### 강화학습
<details markdown="1">
<summary></summary>

주어진 상황에서 어떤 행동을 취할지 **보상** 심리(= 최대효율) 및 *Greedy algorithm*으로 학습한다.

</details>

#### Generative Model(생성모델) vs. Discriminative Model(분류모델)
<details markdown="1">
<summary></summary>

![image](https://user-images.githubusercontent.com/39285147/181839034-973411b8-ce34-49a5-a342-43c51df082b3.png)

### Generative Model
![image](https://user-images.githubusercontent.com/39285147/181838538-56cdb2b8-d561-4eaa-9db2-47bf3ecba85f.png)

생성모델은 주어진 학습 데이터를 학습하여 **학습 데이터의 분포를 따르는** 유사한 데이터를 생성하는 모델로써, 학습 데이터의 분포를 학습하는 것이 생성모델에서 가장 중요하다.

분별모델과 달리 x가 발생할 확률인 P(x)나 카테고리 y에서 x가 발생할 확률 P(x|y)를 명시적으로 계산한다.

이 확률 정보를 이용하여 **새로운 샘플**을 생성할 수 있다.
 
가령, 자연어 처리에서 한 단어(토큰)가 들어오면 다음에 올 적절한 토큰을 생성하는 언어 모델이 하나의 예시이다.

생성모델의 한 예시로는 GAN(Generative Aadversarial Netwrok)이 있다.
- 특정 사람의 필체를 흉내 낸 글씨를 생성하는 모델, 특정 양식의 그림을 생성하는 모델

### Discriminative Model

샘플의 카테고리만을 예측하는데 관심이 있는 모델로써, x라는 샘플이 있을 때 이 샘플의 카테고리가 y 일 확률, 즉 사후 확률 P(y|x)만을 추정하면 된다.

가령, 카테고리가 4개 존재한다면 소프트맥스(softmax)의 경우와 같이 각 카테고리별 사후 확률인 P(y=1|x), P(y=2|x), P(y=3|x), P(y=4|x)를 구한 후, 사후 확률이 가장 높은 카테고리로 분류한다.

예시로는, 특정 데이터의 카테고리를 분류하는 모델이 하나의 예시이다.

</details>

#### GAN
<details markdown="1">
<summary></summary>

![image](https://user-images.githubusercontent.com/39285147/181840182-63585580-4473-486c-851a-657f1627bfdd.png)

**비지도학습**에 사용되는 머신러닝 프레임워크의 한 종류로, 생성자와 구분자가 **서로 대립하며(Adversarial:대립하는)** 서로의 성능을 점차 개선해 나가는 쪽으로 학습이 진행하여 **그럴 듯한 가짜를 만들어내는** 것이 주요 개념이다.

Cost 함수로 **Discriminator Function**을 사용한다.

> **Discriminator**: fake image = 0, real image = 1로 출력하도록 학습하느 과정으로, 생성자 구분자가 서로 번갈아가며 학습을 진행한다.

</details>

****
# *Deep Learning*
#### [Convolution](https://github.com/hchoi256/ai-boot-camp/blob/main/ai/deep-learning/cnn.md)
<details markdown="1">
<summary></summary>
이미지에서 feature를 뽑기위해 사용하는 합성곱 연산 과정이다.
</details>

#### [ANN vs. CNN](https://github.com/hchoi256/ai-boot-camp/tree/main/ai/deep-learning)

#### RNN (순환신경망)
<details markdown="1">
<summary></summary>

![image](https://user-images.githubusercontent.com/39285147/181836934-d0c2970c-6048-4f9b-8a60-a31bbe7a901b.png)

RNN은 히든 노드가 방향을 가진 엣지로 연결되어 순환 구조를 이루는 인공신경망의 한 구조이다.
- 가령, L이 2번연속나왔을때는 O라는 output을 주도록해라~

반복적이고 순차적인 데이터(Sequential data)학습에 특화된 인공신경망의 한 종류로써 내부의 **순환구조**가 들어있다는 특징을 가지고 있다.
- 순환구조: 과거의 학습을 Weight를 현재 학습에 반영한다.

따라서, 과거 정보를 기억하여 활용한다는 점에서 자연어 처리 및 시계열에서 좋은 성능을 보인다.

기존의 *지속적이고 반복적이며 순차적인* 데이터학습의 한계를 해결하는 알고리즘이다 (= 중복되는 은닉층 겹겹이 쌓는 것 그만하고 순환시킨다).

![image](https://user-images.githubusercontent.com/39285147/181836703-d90b20e1-a4a7-4ae8-b01f-1cf047775555.png)

가령, 자연어 처리에서 주어인 'I'가 왔기 때문에 그 뒤는 동사일 것이라고 자연스럽게 예측했고, 전치사 'at'이 왔기 때문에 그 뒤는 명사가 올것이라고 추론하는 과정을 수학적으로 모델링한 것이 바로 RNN이다.

LSTM 모델이 RNN의 한 예시이며, 기존 RNN의 문제점인 장기 의존성(Long-Term Dependency)을 해결하고자 고안되었다.
- 4가지 모듈을 탑재 - 망각 게이트 --> 입력 게이트 --> 출력 게이트
  - 망각 게이트: 과거 정보 버릴지 결정
  - 입력 게이트: 저장된 정보들 이동하며 입력 게이트에서 현재 정보 저장할지 결정
  - 출력 게이트: 어떤 값 출력할지 결정
  - tahn 활성화 함수

> **장기 의존성**: 다루는 데이터의 양이 클 경우 과거의 정보를 기억하는데 한계를 가지기 때문에 힘들다는 한계점을 가지고 있다.

</details>

#### TensorFlow vs. PyTorch vs. Keras
<details markdown="1">
<summary></summary>

![image](https://user-images.githubusercontent.com/39285147/180655690-5eefb525-800b-440f-b3ef-be0f143f3ecd.png)

Tensorflow
- 정적 그래프 생성; 모델 전체 계산 그래프 정의한 다음 ML 모델 실행

PyTorch
- 동적 그래프; 동작 중에 그래프 정의/조작

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

#### 모델 일반화(Model Generalization) 성능 개선 방법
<details markdown="1">
<summary></summary>

1. 새로운 데이터

2. 이미지 확대, 증대, 회전, 밝기 조절, etc.

3. etc. (무궁무진함)

</details>


#### Internal Covariate Shift
<details markdown="1">
<summary></summary>

![image](https://user-images.githubusercontent.com/39285147/182007099-62500132-26ad-4440-bc98-0e200225f89d.png)

**Covariate shift(공변량 변화)**는 공변량 변화라고 부르며 입력 데이터의 분포가 학습할 때와 테스트할 때 다르게 나타나는 현상을 말한다.

**Internal Covariate Shift**는 매 스텝마다 hidden layer에 입력으로 들어오는 데이터의 분포가 달라지는 것을 의미하며 Internal Covariate Shift는 layer가 깊을수록 심화될 수 있다.

역전파의 가중치 업데이트 과정에서 ReLU나 규제화, 학습률을 낮추는 등의 방법으로도 레이어 수가 많아질수록 학습이 잘 되지않는 근본적인 문제점이 존재했다.

이러한 근본적 문제가 바로 Internal Covraite Shift이며, 이는 [**Batch Normalization*](#Batch-Normalization) 기법으로 해결 가능하다.

</details>

#### Normalization, Whitening
<details markdown="1">
<summary></summary>

#### Normalization
![image](https://user-images.githubusercontent.com/39285147/182007273-ff44427f-4bdf-4b7a-8237-27c3dfd16964.png)

데이터를 동일한 범위 내의 값을 갖도록 하는 기법으로 대표적으로 Min-Max, Standardization이 있다. 이 중에서 Standardization은 데이터를 평균 0, 표준편차 1이 되게 변환하여 정규화시킨다.

#### Whitening
![image](https://user-images.githubusercontent.com/39285147/182007309-fdc73544-4902-4661-8191-bdc8aed4f2e4.png)

데이터의 평균을 0, 그리고 공분산을 단위행렬로 갖는 정규분포 형태로 PCA를 사용하여 변환하는 기법이다.

### 한계
whitening을 하게되면 이전 레이어로부터 학습이 가능한 parameters의 영향을 무시한다.

따라서, [**Batch Normalization*](#Batch-Normalization) 방법이 필요하다.

</details>

#### Batch Normalization vs. Layer Normarlization
<details markdown="1">

#### Batch normalization
![image](https://user-images.githubusercontent.com/39285147/182007608-a8c03859-9a8a-451b-bf39-4ebc9876fdc0.png)

CNN에 많이 사용하는 기법으로, 입력 데이터에 대하여 각 차원(feature)별로 mini-batch를 만들어 그 평균과 분산을 계산하는 normalization을 수행한다.

BN은 학습 가능한 parameters가 존재하는 하나의 레이어 구조가 되며 이 기법이 발표된 이후 기존의 딥러닝 구조에서 Convolution Layer와 Activation Layer 사이에 BN Layer가 들어간 형태로 발전했다.

#### Layer normalization

RNN에 많이 사용하는 기법이다.

<summary></summary>
</details>

#### [Support Vector Machine & Margin 최대화 이유?](https://github.com/hchoi256/ai-boot-camp/blob/main/ai/machine-learning/supervised-learning/classification/svm.md)

#### [Optimizer](https://github.com/hchoi256/lg-ai-auto-driving-radar-sensor/blob/main/supervised-learning/gradient-discent.md)

#### [Gradient Descent & Stocastic Gradient Descent](https://github.com/hchoi256/lg-ai-auto-driving-radar-sensor/blob/main/supervised-learning/gradient-discent.md)

#### Local Optima에 빠져도 딥러닝 학습에 긍정적인 이유?
<details markdown="1">
<summary></summary>

실제 딥러닝에서 로컬 옵티마 빠질 확률이 거의 없으며, 실제 딥러닝 모델에서는 수많은 w(가중치)가 존재하여 모든 가중치가 모두 로컬 옵티마라도 빠져서 가중치 업데이트가 종료되야 학습이 종료되기 때문이다.

</details>

#### [Ensemble](https://github.com/hchoi256/lg-ai-auto-driving-radar-sensor/blob/main/supervised-learning/ensemble.md)

#### Fully Connected Layer(= hidden layer)
<details markdown="1">
<summary></summary>

발견한 특징점을 기반으로 이미지를 분류하는 레이어 구간

</details>

#### [BERT vs. GPT](https://hchoi256.github.io/nlp/nlp-basic-transformer/)

#### Batch and Epoch
<details markdown="1">
<summary></summary>

![image](https://user-images.githubusercontent.com/39285147/179927500-1f89d8b9-f1d0-409d-b935-d54028e00113.png)

**Batch**: 전체 트레이닝 데이터 셋을 여러 작은 그룹으로 나누었을 때, 하나의 소그룹에 속하는 데이터 수를 의미한다.
- Batch 사이즈 ↑, 한 번에 처리해야할 양 ↑, 학습 속도가 ↓, 메모리 부족
- Batch 사이즈 ↓, 적은 샘플을 참조해서 가중치 업데이트가 빈번하게 일어나기 때문에, 비교적 *불안정하게* 훈련될 수도 있습니다.

**Epoch**: 딥러닝에서는 epoch은 전체 트레이닝 셋이 신경망을 통과한 횟수이다. 가령, 1-epoch는 전체 트레인이 셋이 하나의 신경망에 적용되어 순전파와 역전파를 통해 신경망을 한 번 통과했다는 것을 의미한다.

</details>

#### Gradient Vanishing/Exploding/Clipping
<details markdown="1">
<summary></summary>

**Gradient Vanishing**: 역전파 과정에서 입력층으로 갈 수록 기울기(Gradient)가 점차적으로 작아지는 현상 (시그모이드 대신 ReLU 사용)

**Gradient Exploding**: 기울기가 점차 커지더니 가중치들이 비정상적으로 큰 값이 되면서 결국 발산된다.

**Gradient Clipping**: 기울기 폭주를 막기 위해 임계값을 넘지 않도록 값을 자른다.

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

#### Pre-training vs. Fine-tuning
<details markdown="1">
<summary></summary>

**Pre-training**
- 기존 임의의 값으로 초기화하던 모델의 가중치들을 다른 문제(task)에 학습시킨 가중치들로 초기화하는 방법이다.
  - 텍스트 유사도 예측 모델을 만들기 전 감정 분석 문제를 학습한 모델의 가중치를 활용해 텍스트 유사도 모델의 가중치로 활용하는 방법이다.

**Fine-tuning**
- 기존에 학습되어진 모델을 기반으로 아키텍쳐를 새로운 목적에 맞게 변형하고 이미 학습된 모델 Weights로 부터 학습을 업데이트 하는 방법이다.
  - 사전 학습 방법인 감정 분석 문제에 사전 학습시킨 가중치와 더불어 텍스트 유사도를 위한 부가적인 가중치를 추가해 텍스트 유사도 문제를 학습하는 것이 미세 조정 방법이다.

</details>

#### Zero-shot vs. One-shot vs. Few-shot Learning
<details markdown="1">
<summary></summary>

**Zero-shot**
- 일반적으로 딥러닝은 training에 사용된 class만을 예측할 수 있다. 따라서 unseen data가 입력되면 seen class로 예측하는 바보가 되버리는데, Zero shot은 train set에 포함되지 않은 unseen class를 예측하는 분야이다.
  - Unseen data를 입력 받아도, seen data로 학습된 지식을 전이하여 unseen data를 unseen class로 예측할 수 있다.
- 모델이 바로 downstream task에 적용한다.

> **CV**: 클래스 레이블 간의 표현 유사성에 의존
>
> **NLP**: 동일한 의미적 공간에서의 레이블을 나타내는 '라벨 이해' 기능 기반


**One-shot**
- 모델을 1건의 데이터에 맞게 업데이트한다.
- 보통의 얼굴 인식 시스템은 사용자의 사진이 한 장만 존재할 수도 있기 때문이다.

**Few-shot**
- 모델을 몇 건의 데이터에 맞게 업데이트한다

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

*Beta*(베타분포): 두 매개변수 α 와 β 에 따라 [0, 1] 구간에서 정의 되는 **단일** 확률변수에 대한 연속확률분포이다.

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

#### Confidence Interval(신뢰구간)
<details markdown="1">
<summary></summary>
모수가 속할 것으로 기대되는 범위 (모수를 포함할 것으로 추정한 구간)
</details>

#### covariance(공분산)/correlation(상관계수)
<details markdown="1">
<summary></summary>

**공분산**: 두 개의 확률변수의 상관정도(**어떻게** 퍼져있는지)를 나타내는 값이다 [-1, 1]. 공분산의 크기는 두 확률변수의 scale에 크게 영향을 받는다.

**상관관계**: 두 변수 간에 선형 관계의 정도를 수량화하는 측도이다. 이때 두 변수간의 관계의 강도를 상관계수(correlation coefficient)라고 한다. 만약, 상관계수가 0이면 두 확률변수는 아무런 선형 상관관계를 갖지 않는다; 양의 선형관계면 1, 음의 선형관계면 -1

공분산 vs. 상관관계
- 공분산: 상관 정도의 절대적인 크기를 측정 X
- 상관관계: 상관 정도의 절대적인 크기를 측정 O

</details>

#### Total Variance
<details markdown="1">
<summary></summary>

![image](https://user-images.githubusercontent.com/39285147/182010388-60cf7098-0c6e-4bc1-b243-cb7c4ff0cd98.png)

두 확률 분포의 측정값이 벌어질 수 있는 가장 큰 값이다.

</details>

#### Explained variation(설명분산) vs Uexplained variation(설명되지 않는 분산)
<details markdown="1">
<summary></summary>

#### Explained variation(설명분산)
통계에서 설명분산은 수학적 모델이 주어진 데이터 세트의 변동(분산)을 설명하는 비율을 측정한다.

**잔차제곱** = 1 – (Sum of Squared Residuals / Total Variance)

**Explained Variance Score** = 1 – ( (Sum of Squared Residuals – Mean Error) / Total Variance )

결정계수(=R제곱)와의 유일한 차이는 SSR에 Mean Error를 빼는 것으로, 모델 학습에 편향성이 존재할 경우 Mean Error가 0이 아닌 값을 가지게 된다.

이 경우, 결정계수과 설명분산점수의 값이 달라지게 되어 편향성 유무를 판별할 수 있다.

일반적으로, 회귀분석과 같은 것들은 잔차에 편향이 없다는 전제로 수행되기에 설명분산점수를 따지지 않는다.

#### Uexplained variation(설명되지 않는 분산)

Unexplained variance는 분산 분석 (ANOVA)에 사용되는 용어로, ANOVA는 다른 그룹의 평균을 비교하는 통계적 방법이다.

The sum of the squared of the differences between the y-value of each ordered pair and each corresponding predicted y-value

</details>

#### Confidence Interval(신뢰구간)
<details markdown="1">
<summary></summary>
모수가 속할 것으로 기대되는 범위 (모수를 포함할 것으로 추정한 구간)
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
모집단의 데이터 분포 비율을 유지하면서 데이터를 샘플링(취득)하는 것을 말한다.

가령, 모집단의 남녀 성비가 각각 54%, 46%라고 한다면 이 모집단에서 취득한 샘플 데이터 역시 남녀 성비가 각각 54%, 46%가 되도록 한다.
</details>

#### Sample Variance를 구할 때, N대신에 N-1로 나눠주는 이유는 무엇인가?
<details markdown="1">
<summary></summary>

**1. 모평균과의 정확도 근사를 위해서**

표본을 무한정 추출하면 표본분산과 표본평균은 모분산과 모평균에 수렴하여야 하지만, 실제로 N으로 나누어 표본분산을 구할 경우 표본분산보다 모분산이 더 큰 현상이 발생한다. 따라서, 표본분산과 모분산의 차이를 줄이기 위하여 표본분산의 크기를 키우고자 분모에 N 대신 N-1을 분모에서 사용한다.

**2. 자유도 (Degree of Freedom)**

분산은 편차 제곱의 평균이므로, 표본평균을 알고 있다는 전제로 도출할 수 있는 값이다. 따라서, 편차 제곱의 평균을 구할 때 분모에 N 대신 N-1을 사용하면, 우리는 표본평균을 알고있기 때문에 마지막 추정값을 더 나은 통계치 도출을 위해 자유롭게 제외할 수 있다. 이를 토대로, 우리는 표본분산의 자유도가 N-1임을 알 수 있다.

</details>

#### [MLE(최대우도법)](https://github.com/hchoi256/ai-terms/blob/main/mle.md)


####  Conjugate Prior
<details markdown="1">
<summary></summary>

사후확률이 사전확률과 동일한 함수형태를 가지도록 해준다.

베이즈 확률론에서 사후확률을 계산함에 있어 사후 확률이 사전 확률 분포와 같은 분포 계열에 속하는 경우 그 사전확률분포는 켤레 사전분포(Conjugate Prior)이다.

</details>

#### Confusion Matrix(FN, TN, TP, FP) with precision and recall
<details markdown="1">
<summary></summary>

![image](https://user-images.githubusercontent.com/39285147/180647570-40aff0f1-6267-4236-8c25-b0226710069d.png)

1. TP (True Positive): 양성(긍정적 결과)이라고 예측한 것이 정답일 때
- 즉 실제로 Positive인 것을 잘 맞췄음

2. TN (True Negative): 음성(부정적 결과)이라고 예측한 것이 정답일 때
- 즉 실제로 Negative인 것을 잘 맞췄음

3. FP (False Positive): 양성(긍정적 결과)이라고 예측한 것이 오답일 때
- 즉 실제 Negative인 것을 Positive로 예측해서 틀렸음

4. FN (False Negative): 음성(부정적 결과)이라고 예측한 것이 오답일 때
- 즉 실제 Positive인 것을 Negative로 예측해서 틀렸음

**Precision Recall Curve**
![image](https://user-images.githubusercontent.com/39285147/182006764-339fc132-96c4-4201-854a-f4513df2b3d5.png)

- x축을 recall, y축을 precision으로 하는 커브를 의미한다.
</details>

#### Frequentist vs. Beyas vs. Naive Beyas(나이브 베이스)
<details markdown="1">
<summary></summary>

### Frequentist
확률을 객관적으로 발생하는 현상이라 본다 (i.e., 주사위 한 번 굴릴 때 1/6); 참된 확률값은 고정값을 가진다.

각 환자의 병은 독립적이라 판단하여 해당 환자를 직접 검사하여 source of pain을 찾는다.

### Beyas 정리
![image](https://user-images.githubusercontent.com/39285147/180647087-74f8de6e-419b-46ec-ae5b-6dbb0bcf0380.png)
![image](https://user-images.githubusercontent.com/39285147/180647146-ebb0bd48-b944-4f61-a16a-8f516d517fdf.png)

두 확률 변수의 사전 확률과 사후 확률 사이의 관계를 나타내는 정리이다.

### Naive Beyas
현상에 대한 관찰자의 주관적 믿음의 체계로써 판단하고 사전확률을 고려하여 과거의 사건이 현재 사건에 영향을 미칠 수 있다고 생각한다; 참된 확률값을 상수가 아닌 분포, 즉 확률 변수라 여긴다.

각 특징들은 서로 '독립적'이라는 점에서 베이즈 정리와 차이점이 있다.

데이터가 각 클래스에 속할 특징 확률을 계산하는 조건부 확률 기반의 분류 방법이다.

비슷한 증상의 이전 환자의 증상과 결합하여 source of pain을 찾는다.

![image](https://user-images.githubusercontent.com/39285147/180639800-fc63b011-c599-4e24-a0c3-2107c0ec71ff.png)
![image](https://user-images.githubusercontent.com/39285147/180640867-9f304ddd-47e5-4a05-b256-775500db9d49.png)

실제 상황에서는 변수들이 서로 알게모르게 의존되어 있어서 모델 학습에 적용하기 어렵다.

이러한 한계를 탈피하여 여러 '독립변수'들로써 모델 학습에 활용하고자 가정하여 Naive라 이름 붙여졌다.

1. Prior Probability

![image](https://user-images.githubusercontent.com/39285147/180639502-806b47c7-6ace-43df-8f75-e9fdab5defce.png)

전체 데이터 수에서 각 집합 데이터 수의 비율을 구한 값을 '사전확률'로 하여 새로운 데이터가 어디에 분류될지 예측한다.

2. Likelihood (우도)

![image](https://user-images.githubusercontent.com/39285147/180639529-60fadaf3-fe23-47a1-ae7f-39064efd35f8.png)

![image](https://user-images.githubusercontent.com/39285147/180639518-043f4741-4aa0-43ce-8806-41a01ac3099d.png)

새로운 데이터 주변에서 한 원에 속하는 범주 안에 속한 빨간/파란공 비율을 각각 도출하여 더 큰 '우도'를 갖는 집합으로 새로운 데이터를 분류한다.

3. Posterior Probability

![image](https://user-images.githubusercontent.com/39285147/180639613-ba4a1e96-ae57-4735-bc82-82a5a8fb93a5.png)

'사후확률'은 사전확률에 우도를 곱한 값으로, 사후확율을 통하여 최종적으로 새로운 데이터의 분류 집합을 결정한다.

</details>

****
# *Lienar Algebra*

