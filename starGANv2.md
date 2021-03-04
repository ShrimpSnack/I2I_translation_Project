Multi-Domain I2I translation 
- Pix2Pix
- CycleGAN
- DiscoGAN
- MUNIT
- StarGAN

### 한계
- 한정된 domain들 내에서만 translate >> **여러 도메인에 대해서 모두 translate**
- 한번에 1가지 domain에 대해서 translate >> **특정 domain에 대한 여러가지 style에 대해서 translate 가능**


## StarGANv2
하나의 모델로 **여러 도메인의 이미지를 생성**할 수 있는 모델
- starGAN에서는 각각의 도메인에 대해 동일한 변형만 가능했다면 StarGANv2는 domain-specific style code로 변경하여 이미지 생성 뿐 아니라 image to image translation이 가능하도록 함
- ```mapping network```와 ```style encoder```모듈을 추가하였으며 다양한 도메인의 다양한 이미지를 합성할 수 있음

### Key Point
- ```Mapping Network```: 임의의 가우스 노이즈를 스타일 코드로 변환하는 것을 학습
- ```Style Encoder```: 주어진 소스 이미지에서 스타일 코드를 추출하는 것을 학습 

### Framework
![image](https://user-images.githubusercontent.com/72767245/109938523-78d4e180-7d13-11eb-835a-2a24f6fd64e8.png)

- X와 Y를 각각 이미지와 가능한 도메인의 집합
- X에 속하는 이미지 x와 Y에 속하는 임의의 도메인 y가 주어졌을 때 StarGANv2의 목표는 하나의 Generator만으로 이미지 x를 도메인 y의 이미지로 변형하되, 다양한 스타일로 변형할 수 있도록 학습

---

**(A) Generator** : input image x가 들어가면 output으로 G(x,s)가 나옴
- style vector인 s는 AdaIN(Adaptive instance normalization)을 통해 주입
- s는 도메인 y의 스타일을 대표하도록 밑에 나올 mapping network F나 style encoder E에 의해 만들어짐

---

**(B) Mapping Network** : random latent vector z와 domain y가 주어졌을때 Mapping Network인 F는 style vector s = Fy(z) 를 만듦
- domain y를 대표하는 latent vector z 를 style vector s로 mapping 해줌
- F는 다중출력 MLP로 구성됨 

---

**(C) Style Encoder**: image x와 domain y가 주어지면 E는 image x 에서 style information을 추출하는 역할. s = Ey(x)

---

**(D) Discriminator**: D는 다중 출력 Discriminator. D는 각 branch는 이미지 x가 real 인지 fake인지 이진분류할 수 있도록 학습
