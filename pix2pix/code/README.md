facade dataset 다운로드
```
!unzip /content/pix2pix-pytorch/dataset/facades.zip -d/content/data; #압축풀기
```
model.py
```python
def main(config):
  # Init Dataset
  loaders = Loaders(config) 
  # Init model Pix2pix
  solver = Solver(config, loaders)
  # Train model
  solver.train()
```
