---
title: '[SNN Basic Tutorial 5] Leaky Integrate and Fire(LIF) 모델 설명'
date: 2022-08-14 17:22:00 +0900
categories: ['[KOR] SNN Basic Tutorial']
tags: [SNN Basic Tutorial, SNN, Spiking Neural Network, LIF, Leaky Integrate and Fire, python, 뉴로모픽, neuromorphic] # TAG names should always be lowercase
author: euijin_jeong
math: true
toc: true
---

### **SNN Basic Tutorial 목차**

1. [Spiking Neural Network란](https://jinprelude.github.io/posts/SNN-Basic-Tutorial-1-Spiking-Neural-Network%EB%9E%80/)

2. [SNN을 위한 기초 뇌과학](https://jinprelude.github.io/posts/SNN-Basic-Tutorial-2-SNN%EC%9D%84-%EC%9C%84%ED%95%9C-%EA%B8%B0%EC%B4%88-%EB%87%8C%EA%B3%BC%ED%95%99/)

3. [SNN을 위한 회로이론(1): 기초](https://jinprelude.github.io/posts/SNN-Basic-Tutorial-3-SNN%EC%9D%84-%EC%9C%84%ED%95%9C-%ED%9A%8C%EB%A1%9C%EC%9D%B4%EB%A1%A0(1)-%EA%B8%B0%EC%B4%88/)

4. [SNN을 위한 회로이론(2): RC회로](https://jinprelude.github.io/posts/SNN-Basic-Tutorial-4-SNN%EC%9D%84-%EC%9C%84%ED%95%9C-%ED%9A%8C%EB%A1%9C%EC%9D%B4%EB%A1%A0(2)-RC%ED%9A%8C%EB%A1%9C/)

5. **[Leaky Integrate and Fire(LIF) 모델 설명](https://jinprelude.github.io/posts/SNN-Basic-Tutorial-5-Leaky-Integrate-and-Fire-%EB%AA%A8%EB%8D%B8-%EC%84%A4%EB%AA%85/)**

때가 왔습니다! 지금까지 배운 내용을 기반으로 이번 글에서는 SNN의 가장 기초적인 모델인 Leaky Integrate and Fire 모델을 알아보도록 하겠습니다.

## **Leaky Integrate and Fire(LIF) 란**

이전 글 [[SNN Basic Tutorial 3] SNN을 위한 회로이론(1): 기초](https://jinprelude.github.io/posts/SNN-Basic-Tutorial-3-SNN%EC%9D%84-%EC%9C%84%ED%95%9C-%ED%9A%8C%EB%A1%9C%EC%9D%B4%EB%A1%A0(1)-%EA%B8%B0%EC%B4%88/) 에서 "자연현상을 수학적으로 나타낼 수 있도록 가공하는 과정" 을 **모델링**이라고 설명하였습니다. 현재 다양한 목적에 맞게 동물의 뉴런을 모델링 한 다양한 모델이 있습니다. 한 예로 뉴런의 action potential을 가장 현실적으로 모사했다고 평가받는 모델은 바로 **[Hodgkin-Huxley model](https://en.wikipedia.org/wiki/Hodgkin%E2%80%93Huxley_model)** 인데, 이 모델은 1952년에 개발되었으며 실제 오징어의 squid giant axon을 실험적으로 분석하여 만들었습니다.

앞서 소개드렸듯이 이번 포스트에서 다룰 모델은 **Leaky Integrate and Fire(LIF) 모델**입니다. LIF 모델은 뉴런이 가지고 있는 다음의 규칙이 모델링되었습니다:

1. pre-synaptic neurons들의 spike를 종합한다.
2. membrane potential이 threshold voltage를 넘으면 spike를 생성하고 reset voltage로 초기화된다.
3. membrane potential voltage는 지속적으로 누수(Leak)가 일어난다.

LIF 모델은 직관적이고 단순한 만큼 낮은 구현 난이도와 연산량 덕분에 SNN을 사용하는 응용 연구분야에 압도적으로 높은 채택비율을 자랑하는 모델이며(반대로 정확한 뉴런 모델을 원한다면 Hodgkin-Huxley 등 더 많은 선택지가 존재합니다), 동시에 SNN 모델을 공부하며 가장 첫번째로 살펴보는 모델이기도 합니다. 

## **LIF의 모델링: RC회로**

이전 포스트에서 언급했듯이 LIF 모델은 RC회로로 모델링될 수 있습니다. LIF모델의 특징과 규칙을 살펴보며 어떤 특징이 RC회로에 어떻게 대응되는지 살펴보겠습니다.

**1. Pre-synaptic neurons들의 spike를 종합한다.**

Pre-synaptic neurons들의 spike는 외부에서 들어오는 전력으로 볼 수 있으며, 이전 포스트에서의 공급원(혹은 강제응답)에 해당합니다.

**2. Membrane은 전하를 저장한다**

뉴런은 pre-synaptic으로부터 action potential을 통해 전해지는 Sodium ion을 뉴런 내에 저장합니다. 이러한 특징은 전력을 일시적으로 저장해놓는 **커페시터**로 모델링될 수 있습니다.

**3. Membrane potential은 시간이 지남에 따라 Resting Voltage로 돌아간다**

Membrane potential은 action potential에 이해 전압이 높아졌다가도 시간이 지나면 membrane을 통해 이온이 빠져나가며 다시 Resting potential, 혹은 Resting Voltage로 돌아갑니다. 이는 회로에 **저항**을 추가시킴으로써 구현해줄 수 있습니다. 또한 이전 회로에서의 $v(0)$이 바로 resting voltage가 되는데, 때문에 아래 그림과 같이 저희가 배운 RC회로에 Resting voltage가 되줄 전압원을 추가시켜주게 됩니다($V_r$은 resting voltage를 나타냅니다):

![LIF circuit](/assets/img/post/2022-08-14/LIF_circuit.PNG)_LIF circuit_

위 그림과 같이 회로를 구성하게 되면서 화로의 $v(0)$을 저희가 원하는 resting voltage로 설정할 수 있게 되었습니다.

**4. membrane potential이 threshold voltage를 넘으면 spike를 생성하고 reset voltage로 초기화된다.**

이 부분은 회로를 통해서가 아닌 코딩을 통해서 처리해주게 됩니다. 전자회로의 전압응답(혹은 membrane potential)을 지속적으로 주시하며, 전압응답이 threshold voltage를 넘게 되면 post-synaptic neuron에게 spike를 전달한 다음, membrane potential을 reset voltage로 초기화시키는 코드를 추가시킬 것입니다.

### **수치해석을 위한 $y(t, h)$ 구하기**
여느 컴퓨터 시뮬레이션과 마찬가지로 SNN 구현체 또한 오일러 방식(Euler method)을 이용하여 membrane potential을 구합니다. 

오일러 방식이란

$$ y\prime = f(t, y)$$

일때

$$ y_{n+1} = y_n + dt \cdot f(t_n, y_n) $$

와 같이 값을 근사합니다.

여기서 $y$는 membrane potential의 $u$가 될 것이고, $f(t_n, y_n)$은 $\frac{du}{dt}$가 될 것입니다. 저희가 이전 글들을 통해 배웠던 회로이론 지식을 통해 $\frac{du}{dt}$을 구할 수 있습니다.

KCL 법칙에 의하여:

$$ I(t) = I_C + I_R $$

여기서:

$I$ : 시간 $t$에서의 입력 전류

$I_C$: 커페시터에 흐르는 전류

$I_R$: 저항에 흐르는 전류

입니다. 식을 계속 전개하면:

$$ I(t) = \frac{u(t) - u_{rest}}{R} + C\frac{du}{dt} $$

여기서:

$u(t)$ : 시간 $t$에서의 membrane potential

$u_{rest}$: Resting Membrane potential

> 우변 첫 번째 항의 분자가 $u(t)$가 아닌 $u(t) - u_{rest}$ 인 이유는, Resting Voltage를 위해 전압원이 추가됨으로 인해 전위차를 뜻하는 전압의 정의에 의해 $u(t) - 0$이 아닌 $u(t) - u_{rest}$ 가 되었기 때문입니다.
{: .prompt-warning }

식을 $\frac{du}{dt}$에 대해 전개해보면 다음과 같습니다:

$$ C\frac{du}{dt} = -\frac{u(t)-u_{rest}}{R}+I(t) $$

$$ RC\frac{du}{dt} = -[u(t)-u_{rest}] + RI(t) $$

마지막으로 RC를 시정수(Time constant) $\tau$로 치환하여 표기하겠습니다.

$$ \therefore \frac{du}{dt} = -\frac{-[u(t)-u_{rest}]+RI(t)}{\tau}$$

## **파이썬으로 LIF 모델을 구현해보자**

코드의 흐름은 다음과 같습니다:

1.   뉴런 클래스 생성
2.   main 함수 생성
3.   main 함수에서 뉴런 클래스 인스턴스 생성
4.   n초의 시간동안 뉴런에 전류를 가하고 매 순간마다 뉴런의 membrane potential 기록
5.   시간에 따른 membrane potential을 그래프로 시각화하여 출력

먼저 뉴런 클래스를 생성하겠습니다.

### **뉴런 클래스 만들기**
마지막으로 LIF의 membrane potential 공식을 한번 더 살펴보겠습니다.

$$ \frac{du}{dt}=\frac{−[u(t)−u_{rest}]+RI(t)}{\tau_{m}}$$

식에서:

- $\tau_m$ : 뉴런의 시정수(Time constant)를 나타냅니다. 코드에서는 `self.tau` 입니다.

- $\frac{du}{dt}$ : membrane potential의 순간변화량을 나타냅니다.

- $u(t)$ : 시간 $t$ 에서의 membrane potential을 나타냅니다. 코드에서는 `self.membrane_potential` 입니다.

- $u_{rest}$: membrane의 resting voltage를 뜻합니다. 코드에서는 `self.mv_rest` 입니다.

- $R$: membrane의 저항을 뜻합니다. 코드에서는 저항의 역수인 컨덕턴스로 쓰이며 `self.g` 입니다.

- $I(t)$: 시간 $t$ 에서의 입력 전류를 뜻합니다. 코드에서는 `input_current_pa` 입니다.

위 식을  다음과 같이 변형하여 membrane potential의 순간변화량을 구할 수 있습니다:



LIF 모델을 구현하는데에 필요한 변수들은 다음과 같습니다:


*   Membrane potential의 resting voltage [$-65mV$]
*   Membrane potential의 threshold voltage [$-55mV$]
*   Membrane potential의 reset voltage [$-70mV$]
*   시정수 $\tau_{m}$ [$10$]
*   저항 $R$ [0.1]

중괄호에 적힌 수치들은 아래 코드에서 기본값으로 쓰일 수치입니다.
주석과 함께 아래 코드를 분석해보겠습니다.

```python
class LifNeuron:
  def __init__(self, mv_reset = -70, mv_threshold  = -55, mv_rest = -65, tau = 10, g = 10):
    self.mv_reset = mv_reset  # resting voltage, mV
    self.mv_threshold = mv_threshold  # threshold voltage, mV
    self.mv_rest = mv_rest  # rest voltage, mV
    self.tau = tau  # tau
    self.g = g  # conductance, nanosiemens, nS

    self.membrane_potential = self.mv_rest # 현재의 membrane potential을 저장할 변수를 추가하고, rest voltage로 초기화한다.
  
  def step(self, input_current_pa = 0, dt = 0.001):
    """
    delta t 의 크기와 delta t 동안 흐르는 전류의 양을 입력받고, 뉴런의 다양한 값들을 반환한다.
    
    Inputs
    ------
    input_current_pa : 주입되는 전류량. 단위는 pA(피코암페어)이다.
    dt : delta t의 크기. 기본값은 millisecond인 0.001초이다.

    Return
    ------
    infos [Dict]
      - membrane_potential : 현재 membrane potential 값, mV.
      - is_spike : 현재 뉴런의 action potential이 생성되었는지를 booloean 값으로 반환.

    """
    is_spike = False   # 
    self.membrane_potential += (-(self.membrane_potential - self.mv_rest) + self.g * input_current_pa)* dt / self.tau
    if self.membrane_potential >= self.mv_threshold :
      is_spike = True
      self.membrane_potential = self.mv_reset
    infos = {
        "membrane_potential": self.membrane_potential,
        "is_spike": is_spike
    }

    return infos
```

```python
if __name__=="__main__":
  observe_ms = 10000 # millisecond, 10초동안 관찰

  neuron1 = LifNeuron() # 뉴런 생성
  membrane_potential_history = [] # 10초동안의 membrane potential을 기록할 리스트, y축
  dt_history = [i/1000 for i in range(observe_ms)]  # 시간(초), x축
  
  for ms in range(observe_ms):
    infos = neuron1.step(input_current_pa = 500, dt = 0.001)
    membrane_potential_history.append(infos["membrane_potential"])

  plt.plot(dt_history, membrane_potential_history)
  plt.show()

```

위 코드는 깃허브와 Colab 2가지 경로로 업로드하였습니다:

- [깃허브](https://github.com/jinPrelude/SNNTutorial/blob/main/SNNBasicTutorial_LIF(Kor).ipynb)
- [Colab](https://colab.research.google.com/drive/1TqHo4lrdqhLW79KMll4dDzdQ6S_pLlwz?usp=sharing)

## **마치며**
이 글을 마지막으로 **[SNN Basic Tutorial] 시리즈**는 마무리하겠습니다. 이해가 안되시는 부분이나 부가적인 설명이 더 추가되었으면 하는 부분은 거리낌없이 댓글 부탁드리겠습니다. 더 좋은 글을 위해 적극 반영하겠습니다. 아직 예정은 없지만 추후 개인적인 동기부여가 생기게 되면 중급 튜토리얼도 연재해보겠습니다. 
그리고 SNN 공부에 있어서 크게 도움 주신 [이성민](https://linkedin.com/in/sung-min-lee-a35855217)님 감사합니다 😭
여기까지 읽어주신 모든 분들께 감사드립니다!
