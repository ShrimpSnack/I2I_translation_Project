# Instance Normalization

- 이미지 변환에 사용
- (1, H, W)의 feature map
- 평균과 표준편차는 batch와 channel과는 무관하게 각 데이터에 대해서만 normalization됨
  - 각 데이터마다 normalization을 따로함. 심지어 filter들의 종류와도 관계없이 다 따로 normalization 진행


# Layer Normalization
- batch N과 무관하게 평균 & 표준편차를 구함
- 동일한 층의 뉴런간의 정규화
- Mini Batch Sample간의 의존관계 없음 (Batch끼리의 연관관계 없음)
- CNN의 경우 BatchNorm보다 잘 작동하지않음(분류문제)
- Layer Norm은 Batch Norm의 mini-batch 사이즈를 뉴런 개수로 변경
- feautre 차원에서 정규화 진행
- 모든 convoltuion filter들까지 다 같이 합쳐서 정규화를 진행


# Batch Normalization
- Batch 단위로 평균을 구함
- 각 데이터들의 분산을 구함
- scale and shift함
- N, H, W에 대해서만 연산을 진행
- 평균과 표준편차는 channel map C와 무관하게 계산되어 batch N에 대해서 normalization 된다.


![image](https://user-images.githubusercontent.com/72767245/118239855-409d0c80-b4d5-11eb-853e-41673fbaf167.png)
