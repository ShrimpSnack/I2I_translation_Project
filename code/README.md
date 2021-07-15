# Main.py
- dataloader(```data_loader.py```)
- build model (```solver.py```)
- Train (```solver.py```)



---


## Dataloader.py
- Transform - ```RandomHorizontalFlip```은 일반적으로 많이 사용
  - ```CeneterCrop```과 ```Resize```부분은 여러개의 Dataset이 필요
  - ```CelebA```: 178 x 218
    - CelebA는 custom dataset을 이용
    - Dataset을 생성할때 Preprocessing 과정이 동반
  - ```RaFD```: 256 x 256
     - torchvision의 ImageFolder를 이용해서 일반적인 방법으로 Dataset 생성
  - 하나의 네트워크를 이용하려면 input 크기가 동일해야됨
    - 비율과 크기를 맞추기 위해서 center부분을 resize 함
- Dataset 생성
- Dataloader 생성

## Build Model
- Training과 Testing을 위한 Solver class를 이용해서 instance를 생성하는 과정(initialize)에서 model build함

```python
solver = Solver(celeb_loader, rafd_loader, config)
```

```generator```의 input으로 input image와 함께 domain 정보를 이용  
```discriminator```에서 domain을 예측하는 Classifier 추가함으로써 단일 generator로 다양한 도메인간 변환이 가능

```python
class Solver(object):
  def __init__(self, celeb_loader, rafd_loader, config):
    # Build the model and tensorboard.
    self.build_model()
    
  def build_model(self):
    self.G = Generator(self.~)
    self.D = Discriminator(self.~)
    
    self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
    self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
```

```
CelebA :
178x218 > 178x178 > 128x128 >> 40 labels

RaFD : 
256x256 > 178x178 > 128x128 >> 8 labels

```

## Generator(CycleGAN의 generator)
- starGAN의 generator부분은 cycleGAN의 architecture을 채택
  - input에서 channel의 경우 image와 domain 정보가 channel-wise로 concat되기 때문에 domain의 개수가 input의 channel 개수에 추가
- stargan은 affine을 True로 사용, cycleGAN에서는 instance norm의 affine을 False로 사용
  - affine이 True이면 output값에 gamma를 곱하고 beta를 더한다.
    - 여기서 gamma와 beta는 learnable parameter이며, beta는 bias 역할을 함
    - 따라서 conv2d에서 bias가 필요가 없음
- ```batch Normalization```을 사용하지 않고 ```instance normalization```을 사용하는 이유는 외견적 invariance을 보존해, style을 변환하는 task에서 좋은 결과를 얻기 위하여
- ```affine``` 사용 >> 통계적 추정치로 인해, style 변환시 품질이 하락할 것임
  - 근데 왜 Affine을 사용????
- ReLU의 inplace는 ```True```
- 최종 layer에서는 normalization된 input image의 값 범위가 -1~1이라 결과값의 범위도 같게 하기 위해서 ```tanh```사용
  - 대부분의 generator에서 tanh 사용

## Discriminator(pix2pix의 patchGAN)
### patchGAN
- ```vanilaGAN```의 경우 output은 real/fake을 예측하는 단일 값
- ```patchGAN```의 경우 output은 16개의 값을 가짐
  - 여러 값을 가진다면 input image의 receptive field가 제한

---


## Network Architecture



![image](https://user-images.githubusercontent.com/72767245/121519560-cce71480-ca2c-11eb-96c9-e0a9be4114cf.png)


![image](https://user-images.githubusercontent.com/72767245/121519581-d5d7e600-ca2c-11eb-9e8d-185865b096ba.png)


## Full Architecture

- **dataLoader**을 생성하는 ```data_loader.py```
- 전체 프로세스 **실행**하는 ```main.py```
- **generator**와 **discriminator**가 정의된 ```model.py```
- model를 **build**하고 **training**하는 ```solver.py```

![image](https://user-images.githubusercontent.com/72767245/121520433-dde45580-ca2d-11eb-848d-6db3907b6588.png)
