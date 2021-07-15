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
