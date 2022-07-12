# Entropy (정보 엔트로피)
![image](https://user-images.githubusercontent.com/39285147/178137939-21da2857-8381-4bae-be75-73744015141d.png)

데이터 집합의 혼잡도를 의미한다. 불확실성(Uncertainty)의 척도로 Entropy가 높다는 것은 정보가 많고 확률링이 낮아 불확실성이 높다.

일반적으로, 분포확률(q)을 아는 경우, 단순 엔트로피로 설명 가능하다.

# Binary Entropy
출력이 0 혹은 1 두 가지 경우만 존재하는 경우 (i.e., OneHotEncoding)

# Cross Entropy
![image](https://user-images.githubusercontent.com/39285147/178137875-eede150d-7788-40c7-937f-4344fa0bb65f.png)
![image](https://user-images.githubusercontent.com/39285147/178137861-a4cb7cb9-1fa0-49f1-967e-957263307c12.png)

분포확률(p)을 모르는 경우, 예측(p) 및 실제확률(q)의 차이인 크로스 엔트로피로 설명 가능하다.

# Kullback-Leibler Divergence (= Relative Entropy)
![image](https://user-images.githubusercontent.com/39285147/178138168-82728f98-d2a5-464b-9b22-6e96e9845a29.png)

서로 다른 두 확률분포에 대한 **다름의 정도**; 두 확률 분포가 동일할 때 KL-Divergence는 0이다.

KL Divergence = Cross Entropy - Entropy
