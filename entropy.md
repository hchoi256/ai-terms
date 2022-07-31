# Entropy (정보 엔트로피)
![image](https://user-images.githubusercontent.com/39285147/178137939-21da2857-8381-4bae-be75-73744015141d.png)

데이터 집합의 혼잡도를 의미한다. 불확실성(Uncertainty)의 척도로 Entropy가 높다는 것은 정보가 많고 확률링이 낮아 불확실성이 높다.

일반적으로, 분포확률(q)을 아는 경우, 단순 엔트로피로 설명 가능하다.

### Information gain
정보이득이란 어떤 속성을 선택함으로 인해서 데이터를 더 잘 구분하게 되는 것을 말한다.

예를 들어, 학생 데이터에서 수능 등급을 구분하는데 있어 수학 점수가 체육 점수보다 변별력이 더 높다고 하자. 그러면 수학 점수 속성이 체육 점수 속성 보다 정보이득이 높다고 말할 수 있다. 

# Binary Entropy
출력이 0 혹은 1 두 가지 경우만 존재하는 경우 (i.e., OneHotEncoding)

# Cross Entropy
![image](https://user-images.githubusercontent.com/39285147/178137875-eede150d-7788-40c7-937f-4344fa0bb65f.png)
![image](https://user-images.githubusercontent.com/39285147/178137861-a4cb7cb9-1fa0-49f1-967e-957263307c12.png)
![image](https://user-images.githubusercontent.com/39285147/179909361-82b8a376-69c9-4669-a564-5b47aecdf122.png)

두 확률 분포의 차이를 구하기 위해서 사용된다.

Cross entropy는 분류 작업에 사용되며 회귀에 주로 사용되는 잔차 제곱보다 더 나은 성능을 제공한다.

분포확률(p)을 모르는 경우, 예측(p) 및 실제확률(q)의 차이인 크로스 엔트로피로 설명 가능하다.

# Kullback-Leibler Divergence (= Relative Entropy)
![image](https://user-images.githubusercontent.com/39285147/178138168-82728f98-d2a5-464b-9b22-6e96e9845a29.png)

서로 다른 두 확률분포에 대한 **다름의 정도**; 두 확률 분포가 동일할 때 KL-Divergence는 0이다.

KL Divergence = Cross Entropy - Entropy
