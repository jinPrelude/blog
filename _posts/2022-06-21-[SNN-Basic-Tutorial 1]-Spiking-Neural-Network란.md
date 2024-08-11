---
title: '[SNN Basic Tutorial 1] Spiking Neural Network란'
date: 2022-06-21 19:53:00 +0900
categories: ['[KOR] SNN Basic Tutorial']
tags: [SNN Basic Tutorial, SNN, Spiking Neural Network, neuro, 뉴로모픽, neuromorphic] # TAG names should always be lowercase
author: euijin_jeong
toc: true
math: true
---

### **SNN Basic Tutorial 목차**

1. **[Spiking Neural Network란](https://jinprelude.github.io/posts/SNN-Basic-Tutorial-1-Spiking-Neural-Network%EB%9E%80/)**

2. [SNN을 위한 기초 뇌과학](https://jinprelude.github.io/posts/SNN-Basic-Tutorial-2-SNN%EC%9D%84-%EC%9C%84%ED%95%9C-%EA%B8%B0%EC%B4%88-%EB%87%8C%EA%B3%BC%ED%95%99/)

3. [SNN을 위한 회로이론(1): 기초](https://jinprelude.github.io/posts/SNN-Basic-Tutorial-3-SNN%EC%9D%84-%EC%9C%84%ED%95%9C-%ED%9A%8C%EB%A1%9C%EC%9D%B4%EB%A1%A0(1)-%EA%B8%B0%EC%B4%88/)

4. [SNN을 위한 회로이론(2): RC회로](https://jinprelude.github.io/posts/SNN-Basic-Tutorial-4-SNN%EC%9D%84-%EC%9C%84%ED%95%9C-%ED%9A%8C%EB%A1%9C%EC%9D%B4%EB%A1%A0(2)-RC%ED%9A%8C%EB%A1%9C/)

5. [Leaky Integrate and Fire(LIF) 모델 설명](https://jinprelude.github.io/posts/SNN-Basic-Tutorial-5-Leaky-Integrate-and-Fire-%EB%AA%A8%EB%8D%B8-%EC%84%A4%EB%AA%85/)


**SNN Basic Tutorial** 연재에서는 딥러닝에 대한 기초적인 이해는 있지만 SNN은 처음이신 분들이 SNN의 기본적인 개념부터, 가장 기초적인 SNN 모델인 Leaky Integrate and Fire model(LIF)의 미분방정식까지 이해하시는 것을 목표로 작성하였습니다. 또한 파이썬을 이용하여 간단한 LIF 모델까지 구현하여 볼 것입니다.

## **SNN 이란**

SNN은 인공신경망(Artificial Neural Network, ANN)의 한 종류입니다. 인공신경망은 동물의 신경망이 정보를 처리하는 방식을 모방하여 만든 네트워크를 일컫는 말입니다. SNN은 이름 그대로 **Spiking**이 정보의 단위가 되며, 인공신경망이 그렇듯 뉴런과 시넵스로 이루어진 네트워크를 통해 전파됩니다. Spike는 기존 딥러닝 네트워크들(MLP, RNN, CNN 등)이 Tensor 혹은 실수값(float)을 주고받는 것에 비해 특정 뉴런에서 특정 시간에 Spike가 발생했는지에 대한 이산적인 정보만을 주고 받습니다.

![Spiking Neurons의 정보처리 방식](/assets/img/post/2022-06-21/showOpenGraphArticleImage.jpg)
_Spiking Neurons의 정보처리 방식, [출저](https://aip.scitation.org/doi/abs/10.1063/1.5042243)_

그림은 두 개의 Spiking Neurons이 연결되어 있으며, pre Neuron이 post Neuron에게 정보를 전달하도록 설정되어있는 상태입니다. pre Neuron의 우측에 그려진 1차원 그래프는 시간축에 따라 pre Neuron이 post Neuron에게 총 3번의 Spike를 보낸다는 점을 알려줍니다. 사진의 가운데 파란색으로 채색된 그래프는 시간에 따른 post Neuron의 **활동전위(action potential)**를 나타냅니다(활동전위에 대해서는 [SNN을 위한 기초 뇌과학](https://jinprelude.github.io/posts/SNN-Basic-Tutorial-2-SNN을-위한-기초-뇌과학/)에서 자세히 다루겠습니다). Post Neuron의 활동전위는 pre Neuron으로부터 Spike를 받을 떄 특정한 값만큼 치솟고, 시간이 지남에 따라 서서히 줄어듭니다(Leaky). 세 번째 Spike가 post Neuron에 들어오게 되면서 활동전위는 임의의 **임계값(threshold)**인 $u_{th}$를 넘게 되고, 그와 동시에 post Neuron의 활동전위의 값은 0으로 초기화됩니다. 그리고 사진의 가장 아랫쪽 빨간색으로 채색된 그래프를 보았을 때 post Neuron의 활동전위가 임계값을 넘는 동시에 post Neuron에서 **Spike가 생성**되었음을 알 수 있습니다.

다수의 Pre Neurons가 연결되어있을 때에도 동일합니다(아래 그림 참고). 시간축에 따라 Pre Neurons의 Spiking을 받고 Post Neuron의 활동전위가 임계값을 넘게 되면 Post Neuron은 Spiking을 생성해 냅니다. 

![다수의 Pre Neurons가 있는 SNN](/assets/img/post/2022-06-21/spikingneural630-810x435-c-default.jpg)
_다수의 Pre Neurons가 있는 SNN, [출저](https://www.eenewseurope.com/en/eta-adds-spiking-neural-network-support-to-mcu/)_


## **SNN의 특징과 장단점**

> **본 글에서는 DNN을 tensor 기반 인공신경망(MLP, RNN, CNN 등)을 통칭하는 용어로 사용하였습니다. 더 좋은 표현이 있다면 반영하겠습니다.**
{: .prompt-warning }

현재 연구적/상업적으로 가장 성공적인 인공신경망은 Deep Neural Network(DNN)기반의 인공신경망들이며, 일상생활에서 접할 수 있는 거의 모든 인공신경망은 DNN 기반입니다. DNN의 발전으로 인해 저희는 지금 손에 쥐어진 핸드폰의 음성비서부터 자율주행자동차까지 불과 몇년 전만 해도 소설에 나올법한 기술들을 일상생활에서 접하고 사용하고 있습니다. 그렇다면 SNN에 대한 연구가 지속되고 있는 데에는 어떠한 이유들이 있을까요?

### **장점 1. 낮은 전력소모**
SNN이 상용화되었을 때 가장 수요가 많은 것으로 예상되는 분야는 바로 Robotics와 Edge computing 시장입니다. 두 분야 모두 low latency와 low energy consumption을 필요로 하는 분야인데, SNN이 기존 DNN보다 **낮은 전력**으로 기동 가능한 특징이 있기 때문입니다. 

무거운 양의 Tensor끼리의 행렬곱을 통해 연산이 이뤄지는 기존 DNN 방식과는 달리 SNN은 Spike의 누적에 의한 활동전위가 임계점을 넘으면 Spike를 보내는 단순한 구조로 되어있습니다. 이러한 방식은 **아날로그적인 방식**으로 구현이 가능하여 무거운 행렬연산을 위한 Computing Unit이 필요한 DNN과 달리 회로 레벨에서 직접적으로 신경망 구동을 설계할 수 있다는 장점이 있습니다. 때문에 SNN이 상용화 될 경우 저전력 고연산 작업이 필요할 것으로 예상되는 Robotics 분야가 특히 수혜를 볼 것이라는 의견이 있습니다. 이러한 특징때문에 SNN은 머신러닝 분야에서보다 반도체 분야에서 오히려 주목하고 SNN의 발전에 미리 준비하고 있는 모습입니다. 대표적으로 SNN신경망에 특화된 Intel의 Loihi 칩이 있습니다.
![Intel의 Loihi칩](/assets/img/post/2022-06-21/intel-loihi-2-1-16x9.jpg)
_SNN에 특화된 Intel의 Loihi칩-[출처](https://www.intel.com/content/www/us/en/newsroom/news/intel-unveils-neuromorphic-loihi-2-lava-software.html)_


### **장점 2. Bio-Plausible**
"Spiking을 이용하여 정보를 전달한다"는 개념은 **실제 동물의 뇌가 그렇게 작동한다**는 점에서 영감을 받았습니다. DNN의
경우 형상학적으로(topologically) 동물의 신경망에서 영감을 받았지만, 정보를 Spiking이 아닌 실수값으로 나타낸다는 점, 신경망을 갱신할 때 경사하강법(Gradient Descent)과 같은 Global Optimizartion Rule을 사용한다는 점에서 Bio-Plausible(생물학적으로 타당)하지 못합니다. 생물학적으로 가장 올바른 SNN 모델인 **Hodgkin-Huxley model**의 경우 머신러닝을 위해서가 아닌 실제 생물 뉴런의 활동전위를 수학적으로 모델링하기 위해 고안된 만큼(두 사람은 이 연구를 통해 1963년 노벨 생리학·의학상을 받았습니다-[출저](https://ko.wikipedia.org/wiki/%EB%85%B8%EB%B2%A8_%EC%83%9D%EB%A6%AC%ED%95%99%C2%B7%EC%9D%98%ED%95%99%EC%83%81))  컴퓨터과학에 뿌리를 두고 있는 DNN과는 달리 SNN은 생물학에 기반을 두고 태어났습니다. 이러한 배경 때문에 DNN이 풀고 있는 문제를 SNN에 적용하는 과정에 있어서 괴리가 생기기도 하지만, 동시에 **동물 뇌의 학습방법에 대한 연구가 발전함에 따라 SNN의 잠재능력 또한 확장**될 수 있다는 특징이 있습니다. 이러한 특징은 동물의 뇌를 컴퓨터로 모델링 및 시뮬레이션하는 분야와  Bio-inspired 분야에 관심이 있는 머신러닝 연구자들로 하여금 관심을 가지도록 만들었습니다(저도 마찬가지고요ㅎㅎ).

### **한계점: 효과적인 학습방법의 부재**
이러한 장점에도 불구하고 여전히 상용화와 거리를 좁히지 못하게끔 하는 SNN의 가장 큰 한계점이 있는데, **아직까지 SNN의 효과적인 학습방법을 찾지 못했다**는 점입니다. **STDP, ReSuMe, Dopamine-modulated STDP** 등 다양한 학습방법이 존재하지만, DNN만큼 SNN을 상용화로 이끌만큼 강력한 학습방법론은 아직 전무한 상태입니다. 아무리 이론적으로 기존의 신경망보다 효율적이고 우월하다고 하더라도 효과적인 학습방법이 없다면 그 가치를 인정받기는 어렵습니다. 이는 DNN도 마찬가지였습니다. 비록  1950년대에 고안되었지만 컴퓨팅 성능, Gradient vanishing, 데이터 부족 등의 이유로 3번의 시대적 비수기를 겪고, 2010년 초가 되어서야 비로소 빛을 본 것과 같이 말이죠. 하지만 위에서 언급했듯이 DNN이 가지고 있지 않은 SNN만의 특징들이 그 메리트가 없어지지 않는 한, SNN을 상용화하기 위한 연구개발은 계속 될 것입니다.

## **마치며**

본 글에서는 SNN의 소개를 다루었습니다. 다음 글부터는 SNN을 이해하기 위한 기초지식들을 다룰 예정입니다. SNN을 이해하기 위한 기초적인 **뇌과학, 회로이론, 미분방정식** 을 먼저 살펴본 다음, SNN에 대해서 본격적으로 다룰 계획입니다.

다음 글인 **[SNN을 위한 기초 뇌과학](https://jinprelude.github.io/posts/SNN-Basic-Tutorial-2-SNN%EC%9D%84-%EC%9C%84%ED%95%9C-%EA%B8%B0%EC%B4%88-%EB%87%8C%EA%B3%BC%ED%95%99/)** 에서는 실제 동물의 뉴런과 시넵스의 구조, 역할에 대해서 알아보도록 하겠습니다.
