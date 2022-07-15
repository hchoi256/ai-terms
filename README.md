****
# *Machine Learning*

### [Entropy](https://github.com/EricChoii/ai-terms/blob/main/entropy.md)

### 계층적샘플링
<details markdown="1">
<summary></summary>
모집단의 데이터 분포 비율을 유지하면서 데이터를 샘플링(취득)하는 것을 말합니다
</details>

### [시계열 분석](https://github.com/EricChoii/ai-terms/blob/main/time-series-analysis.md)
### [Overfitting](https://github.com/EricChoii/ai-terms/blob/main/overfitting.md)
### 차원의 저주
<details markdown="1">
<summary></summary>
차원이 증가하면서 학습데이터 수가 차원 수보다 적어져서 성능이 저하되는 현상
</details>

### 전이학습(Transfer Learning)
<details markdown="1">
<summary></summary>
전이 학습(Transfer Learning)은 특정 분야에서 학습된 신경망의 일부 능력을 유사하거나 전혀 새로운 분야에서 사용되는 신경망의 학습에 이용하는 것을 의미합니다.
</details>

****
# *Deep Learning*
### Convolution
<details markdown="1">
<summary></summary>
이미지에서 feature를 뽑기위해 사용하는 합성곱 연산 과정이다.
</details>

### Fully Connected Layer
<details markdown="1">
<summary></summary>
발견한 특징점을 기반으로 이미지를 분류하는 레이어 구간
</details>

### GAP
<details markdown="1">

![image](https://user-images.githubusercontent.com/39285147/179212153-3e0a7ee0-a531-43ff-a0e4-64254f9aa362.png)

GAP layer는 각각의 feature map의 값들을 평균을 취한 것으로, feature map의 크기와 관계없이 channel이 k개라면 k개의 평균 값을 얻을 수 있다.
- GAP는 FC Layer와 달리 연산이 필요한 파라미터 수를 크게 줄일 수 있다
  - Regulariztion과 유사한 동작을 통해 overfitting을 방지할 수 있다.
- FC layer는 Convolution layer에서 유지하던 위치정보가 손실되는 반면, GAP layer는 위치정보를 담고 있기 때문에 localization에 유리하다.
<summary></summary>
</details>

### Grad_CAM
<details markdown="1">

![image](https://user-images.githubusercontent.com/39285147/179212552-2f937eab-1a74-4fa9-9a98-5e27c0db0f39.png)

어떤 target concept일지라도 final convolutional layer로 흐르는 gradient를 사용하여 이미지의 중요한 영역을 강조하는 localization map을 만든다.
<summary></summary>
</details>


### Back Propagation
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

### Central Limit Theorem
<details markdown="1">
<summary></summary>

#### 정의
모집단(평균: μ, 표준편차: σ)이 어떤 분포를 따르던지 무관하게, 표본평균의 표본분포는 n이 커지면(>= 30) 평균이 μ이고 표준편차가 σ/n인 **정규분포**를 따른다.

#### 의의
표본평균을 통해서 모집단의 모수인 모평균과 모표준편차를 추정할 수 있는 확률적 근거 제시
</details>

### Law of Large Numbers (LLN)
<details markdown="1">
<summary></summary>
경험적 확률과 수학적 확률 사이의 관계를 나타내는 법칙; 표본집단의 크기가 커지면 그 표본평균이 모평균에 가까워짐을 의미
</details>

### Normalization vs. Standardization
<details markdown="1">
<summary></summary>
정규화: [0, 1] 분포

표준화: 평균 0, 표준편차 1 분포
</details>

### [Bias-Variance Trade-off](https://github.com/EricChoii/lg-ai-auto-driving-radar-sensor/blob/main/supervised-learning/sl-foundation.md)

****
# *Lienar Algebra*

