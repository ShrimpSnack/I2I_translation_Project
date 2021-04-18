## ```main.py```

- Train을 수행하여 main.py를 실행해주면 --인자로 모든 인자들을 받아와 config 됨

```python

if config.mode == 'train':
  if cofig.dataset in ['CelebA', 'RaFD']:
    solver.train()
if config.mode == 'test':
  if cofig.dataset in ['CelebA', 'RaFD']:
    solver.train()
```

train 과 test는 ```solver.py``` 내에 저장되어있다.

#### 변수 설명

- ```c_dim``` :데이터셋 에서 사용할 특성의 수
- ```image_size```: [1,3,256,256]
- ```g_conv_dim```: Generator구조의 첫번째 layer의 filter 수 (default = 64)
- ```g_repeat_num```: Generator구조의 Residual Block의 수(default = 6)
- ```d_conv_dim```: Discriminator구조의 첫번째 layer의 filter 수 (default = 64)
- ```d_repeat_num```: Discriminator구조의 Output layer를 제외한 convolution layer의 수 (default = 6)
- ```lambda_gp```: adversarial loss를 구하는데 사용되는 gradient penalty의 값 (default = 10)
- ```num_iters```: 학습 과정에서의 몇번의 iteration을 돌 것인지(default = 200000)
- ```n_critic```: Discriminator가 몇 번 업데이트 되었을 때 Generator를 한번 update시킬 것인지
- ```selected_attrs```: CelebA 데이터셋에서 사용할 특성들 ('Black Hair, Blond Hair, Brown Hair, Male, Young')
- ```test_iters```: 모델 테스트를 위해 학습된 모델을 몇번쨰 step에서 가져 올 것인지
- ```num_workers```: 몇개의 CPU 코어에 할당할 것인지
- ```mode```: 모델을 Train으로 할 것인지 test로 할 것인지


## ```model.py```
#### ```ResidualBlock```
- Generator의 Bottleneck부분에 사용되는 ```residual block```
```python
class ResidualBlock(nn.Module):
  def __init__(self, dim_in, dim_out):
    super(ResidualBlock, self).__init__()
```

- layers [conv-instanceNorm-ReLU-Conv-InstanceNorm]
```python
self.main = nn.Sequential(
  nn.Conv2d(dim_in, dim_out, kernel_size = 3, stride = 1, padding = 1, bias = False),
  nn.InstanceNorm2d(dim_out, affine = True, track_running_stats = True),
  nn.ReLU(inplace = True),
  nn.Conv2d(dim_out, dim_out, kernel_size = 3, stride = 1, padding = 1, bias = False),
  nn.InstanceNorm2d(dim_out, affine = True, track_running_stats = True))
```
```python
def forward(self,x):
  return x + self.main(x)
```

#### ```Generator```
- 첫번째 Convolution layer의 output dimension인 ```conv_dim```, domain label 수 ```c_dim```, Residual Block의 수 ```repeat_num```
```python
class Generator(nn.Module):
  def __init__(self, conv_dim = 64, c_dim = 5, repeat_num = 6):
    super(Generator, self).__init__()
```

layers이라는 list 안에 layers.append을 해줌
- 첫 conv2d(3 depth + c_dim) 해주어야 함 (depth: 8 -> 64)

```python
layers = []
layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size = 7, stride = 1, padding = 3, bias = False))
layers.append(nn.InstanceNorm2d(conv_dim, affine = True, track_running_stats = True))
layers.append(nn.ReLU(inplace = True))
```
**DownSampling Layer**

```python
curr_dim = conv_dim #64dim

# [conv-instanceNorm-ReLU] 2번 반복
for i in range(2): 
  layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
  layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
  layers.append(nn.ReLU(inplace=True))
  curr_dim = curr_dim*2
