# pix2pix
파라미터를 통한 이미지 생성이 아닌 **이미지로부터 이미지 생성** 하는 모델  

```pix2pix```는 CGAN(conditional GAN)의 한 종류  
CGAN에서는 '조건 **벡터**와 이미지의 짝'을 학습데이터로써 그 대응 관계를 학습    
pix2pix는 '조건 **이미지**와 이미지의 짝'을 학습데이터로써 그 대응 관계를 학습    

## loss function
### Loss1
y와 G(x)간의 차이  
blur 처리가 됨  -> generator는 어느것을 택해도 loss가 너무 커지지 않도록 중간의 어떤 애매한것을 택하는 경향을 보임(Average 선택)  

### Loss2
- 생성된 이미지를 좀 더 실제 같이 만들기
Discriminator 와 Generator가 서로 경쟁하여 이미지가 뿌옇게 나오는 현상을 없앰  
이때 cGAN의 모델을 사용함  
- cGAN: 특정 조건(condition)을 부여하여 학습  
![image](https://user-images.githubusercontent.com/72767245/132973619-54ed8d55-a915-4122-8b3a-a3d42626ad48.png)

- GAN: 어떤 데이터 분포를 입력받아 실제에 가깝게 데이터를 생성하는 모델  
  - 주어진 이미지와 구분하기 힘든 그저 진짜같은 새로운 이미지를 생성
- cGAN: 조건으로 input image을 넣어준다면 입력 이미지가 조건이 되는 것
  - 입력이미지와 '연관된' 이미지를 생성할 수 있게 되는 것
[random noise vector: z, output: y, input: x]    
[GAN: 랜덤 노이즈 벡터 z 에서 출력 이미지 y로 mapping/ G:z-> y]  
[cGAN: 관촬된 이미지 x와 랜덤 노이즈 벡터 z를 y로 학습/ G:x,z -> y]

### 최종 Loss Function
- L1 loss 가 L2 loss 보다 흐린 이미지 처리에 더 좋았음
![image](https://user-images.githubusercontent.com/72767245/132973727-d6de9276-4992-4686-a540-abaf5a224f7e.png)  
- cGAN의 loss function
![image](https://user-images.githubusercontent.com/72767245/132973742-a3d1eb57-8125-4f48-a439-7b006d0d7c62.png)
- 최종 loss function
![image](https://user-images.githubusercontent.com/72767245/132973751-e93b316e-d3fc-4d6d-bace-33b5b57cf4f5.png)

- cGAN에 L1loss을 합친 이유: cGAN에서 input x를 참고한다 하더라도 generator G의 궁극적인 관심사는 D를 속이는 것이기 때문에 상대적으로 x가 덜 반영될까봐


## 네트워크 구조
- Generator의 기본 구조는 **U-Net**
  - U-Net은 Encoder-Decoder의 skip connection 추가한 형태
    - skip connection은 decoder가 잘 학습되지 않는 것을 해결해 줄 수 있음
- Discriminator의 기본구조는 **Patch-GAN**
  - Patch에 대해서 참/거짓 판별한 뒤, 참이 많으면 참, 거짓이 많으면 거짓
  - 더 지역적인 특징이 반영되므로, high-frequency에서 구분하는데 적절
  - N이 작을수록, 전체 매개변수의 수가 작아지므로, 학습속도는 빨라짐
  - Patch의 갯수가 늘어날 수록 선명도가 상승

---
참고자료  
https://di-bigdata-study.tistory.com/8
