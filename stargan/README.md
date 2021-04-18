## StarGAN
[StarGAN - Official PyTorch Implementation](https://github.com/yunjey/stargan)

[StarGAN - paper](https://arxiv.org/abs/1711.09020)

### Generator
- G는 input으로 image와 동시에 target domain label을 받고 fake image을 생성
- G는 original domain label을 가지고 fake image를 다시 original image로 reconstruction하려고 함

```Target domain / Input Image``` -> ```G``` -> ```Fake Image```   
```Original domain / Fake Image``` -> ```G``` -> ```Reconstructed Image```

### Discriminator
- D는 real image와 fake image을 구별하는 것과 동시에 real image일 때 그것과 상응하는 domain을 분류해내는 것을 학습

```Real image / Fake image``` -> ```D``` -> ```Real/Fake``` & ```Domain Classification```

![image](https://user-images.githubusercontent.com/72767245/115122583-99bb7280-9ff3-11eb-8ecc-f010c817b4c7.png)

