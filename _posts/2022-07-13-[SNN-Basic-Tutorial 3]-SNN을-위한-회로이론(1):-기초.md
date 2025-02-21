---
title: '[SNN Basic Tutorial 3] SNN을 위한 회로이론(1): 기초'
date: 2022-07-13 11:01:00 +0900
categories: ['[KOR] SNN Basic Tutorial']
tags: [SNN Basic Tutorial, SNN, Spiking Neural Network, circuit, capacitor, 뉴로모픽, neuromorphic] # TAG names should always be lowercase
author: euijin_jeong
math: true
toc: true
---

### **SNN Basic Tutorial 목차**

1. [Spiking Neural Network란](https://jinprelude.github.io/blog/posts/SNN-Basic-Tutorial-1-Spiking-Neural-Network%EB%9E%80/)

2. [SNN을 위한 기초 뇌과학](https://jinprelude.github.io/blog/posts/SNN-Basic-Tutorial-2-SNN%EC%9D%84-%EC%9C%84%ED%95%9C-%EA%B8%B0%EC%B4%88-%EB%87%8C%EA%B3%BC%ED%95%99/)

3. **[SNN을 위한 회로이론(1): 기초](https://jinprelude.github.io/blog/posts/SNN-Basic-Tutorial-3-SNN%EC%9D%84-%EC%9C%84%ED%95%9C-%ED%9A%8C%EB%A1%9C%EC%9D%B4%EB%A1%A0(1)-%EA%B8%B0%EC%B4%88/)**

4. [SNN을 위한 회로이론(2): RC회로](https://jinprelude.github.io/blog/posts/SNN-Basic-Tutorial-4-SNN%EC%9D%84-%EC%9C%84%ED%95%9C-%ED%9A%8C%EB%A1%9C%EC%9D%B4%EB%A1%A0(2)-RC%ED%9A%8C%EB%A1%9C/)

5. [Leaky Integrate and Fire(LIF) 모델 설명](https://jinprelude.github.io/blog/posts/SNN-Basic-Tutorial-5-Leaky-Integrate-and-Fire-%EB%AA%A8%EB%8D%B8-%EC%84%A4%EB%AA%85/)

지난 글에서는 SNN을 이해하는데 최소한으로 필요하다고 생각하는 기초적인 뇌과학을 다루었습니다. 이번 포스트에서는 SNN을 이해하는데 도움이 될 기초 회로이론을 다뤄보겠습니다.

어떠한 자연 현상을 컴퓨터로 시뮬레이션하기 위해서는 그 현상을 수학적으로 나타낼 수 있도록 가공하는 과정이 필요한데, 저희는 이러한 과정을 **모델링**이라고 부릅니다. 시뮬레이션에는 모델링 과정이 필수적인 과정이고, SNN 또한 뉴런의 action potential을 수학적으로 모델링한 결과라고 볼 수 있습니다. 그럼 연구자들은 SNN을 어떻게 모델링하였을까요? 연구자들은 뉴련의 action potential을 모델링하기 위해 뉴런을 전기회로에 대입하여 생각하였습니다. 실제로 뉴런이 전기신호를 주고받기 때문에 전기회로로 모사하기 매우 적합했기 때문입니다. 저자가 처음 SNN을 공부할 때 SNN의 모델 공식이 잘 와닿지 않았는데, 전기회로에 대한 매우 기초적인 지식만 알고 가도 이해하는데 훨씬 수월하였습니다. 때문에 간단하게라도 기초적인 회로이론을 다루고 넘어가면 입문자 분들께 도움이 되지 않을까 하여 이번 글을 작성하게 되었습니다. 조금이라도 회로에 대한 기초적인 지식이 있으시거나 미분방정식이 익숙하신 분들은 바로 [다음 글](https://jinprelude.github.io/blog/posts/SNN-Basic-Tutorial-5-Leaky-Integrate-and-Fire-%EB%AA%A8%EB%8D%B8-%EC%84%A4%EB%AA%85/)을 읽어보시는 것도 추천드립니다. 공식이 낮설거나 부담스럽게 느껴지신다면, 이 글을 읽으신 다음 다시 다음 글을 읽어보시면 더욱 수월하게 이해하실 수 있으실 겁니다.


## **전기**
일상생활과 가장 밀접한 물리법칙은 거리, 속도, 가속도가 아닐까 싶습니다. 가속도가 누적되어 속도가 되고, 속도가 누적되어 이동거리의 변화로 이어집니다. 그럼 회로를 지배하는 물리법칙은 무엇일까요? 바로 전기입니다. 이번 **전기** 단원을 통해 전기와 관련된 기초 상식을 먼저 살펴보고 가겠습니다.

### **전하**
전기의 가장 기본적인 단위는 **'전하'** 입니다. 전하란 전기적 성질의 크기를 말하는데, 예를 들어 전기적으로 양성(+)을 띄는 양성자는 양전하를 띈다고 말합니다. 전하의 단위는 C(콜룸)이며, $6.24 \cdot 10^{18}$ 개의 양성자가 갖는 전하를 1C로 정의합니다. 반데로 전기적으로 음성(-)을 띄는 전자가 $6.24 \cdot 10^{18}$개 있을 경우 -1C로 정의할 수 있습니다. 수식에서 전하는 Q 혹은 q로 나타냅니다.

### **전류**
 전류란 단위시간동안 한 점을 통과하는 전하의 양을 의미하며, 식으로 나타내면 다음과 같습니다:

$$ I = \frac{\Delta q}{\Delta t} $$

 단위는 A(암페어)이며, 1A는 1초동안 한 점을 통과하는 전하의 양이 1C임을 의미합니다.

### **전압**
전압은 전위의 차이를 뜻합니다. 때문에 전위가 무엇인지 먼저 알아보겠습니다. 전위는 단위전하가 갖는 전기적 포텐셜 에너지를 뜻합니다. 말이 조금 어려우시다면 위키피디아에서 가져온 아래 그림을 참고하시기 바랍니다. 아래 그림에서 큰 구는 1C의 전하를, 작은 구는 -1C의 전하를 뜻하며, 보라색은 전위가 가장 높은 부분을, 청록색은 전위가 가장 낮은 부분을 뜻합니다. 전위는 양전하로부터 거리가 가까울 수록 높아지며, 음전하와 가까워질 수록 낮아집니다. 그리고 전하와 거리가 무한이 멀어졌을 때 전하가 0으로 수렴한다고 정의합니다.

 ![Electric potential](/assets/img/post/2022-07-13/VFPt_metal_balls_largesmall_potential+contour.svg.png)_electric potential, [출저](https://en.wikipedia.org/wiki/Electric_potential)_

전압이란 기준점 A와 비교점 B의 전위의 차이를 뜻하며, 단위는 V(볼트)입니다. 때문에 전압은 절대적인 값이 아닌 상대적인 값임을 알 수 있으며, 기준점과 비교점 모두 존재해야 전압이 성립합니다. 그 예로 아래 배터리 예제를 살펴보겠습니다. 두 베터리는 각각 다른 음극과 양극의 전압을 가지고 있지만, 두 베터리의 전압은 10V로 같습니다. 왜냐하면 두 베터리의 양극을 기준으로 봤을 때 음극과의 전압차이가 10V로 같기 때문입니다.

![batteries](/assets/img/post/2022-07-13/batteries.PNG)_전압 예제_

## **법칙들**
위 단원에서 회로를 이해하기 위한 기본적인 개념들인 전하, 전위, 전류, 전압에 대하여 알아보았습니다. 이번 단원에서는 회로에 있어서 가장 기초적인 법칙 3가지를 알아보도록 하겠습니다.

### **옴의 법칙(Ohm's Law)**
옴의 법칙은 대부분 배우신 기억이 나실 겁니다. 옴의 법칙은 다음과 같이 나타냅니다:

$$ V = IR $$


여기서 V는 전압, I는 전류, R은 저항입니다(저항은 전류를 방해하는 정도이며, 단위는 $\Omega$(옴), 표기는 R입니다). 옴의 법칙을 시각적으로 이해하기 위해 가장 간단한 회로를 가져왔습니다.

![ohm's law](/assets/img/post/2022-07-13/ohmslaw.jpg){: width="200" height="280"}_Ohm's Law, [출저](https://ko.wikipedia.org/wiki/%EC%98%B4%EC%9D%98_%EB%B2%95%EC%B9%99)_

위 회로는 저항만 달려있는 매우 단순한 형태의 회로입니다. 그리고 전압 V가 존재하며 이로 인해 전류 I 가 생성되고 있습니다. 또한 그림에서 알 수 있는 사실은 전류는 전위가 높은 +에서 -로 흐른다는 사실입니다. 

저희는 옴의 법칙을 이용하여 V, I, R 3 가지 변수 중 두 가지만 알면 나머지 하나의 값을 구할 수 있습니다. 예를 들어 I가 3A, R이 $1\Omega$일 경우 V는 $3A \cdot 1\Omega = 3V$가 됨을 알 수 있습니다. 다른 예로 V가 5V, R이 $2\Omega$일 경우 I는 $I = \frac{5V}{2\Omega} = 2.5A$ 가 됨을 알 수 있습니다.

### **키르히호프의 법칙(Kirchhoff's Law)**
전기회로에서 가장 기초가 되는 법칙입니다. 키르히호프의 법칙에는 2가지가 있습니다:

**키르히호프의 전류 법칙(Kirchhoff's Current Law)** :

![Kirchhoff's Current Law](/assets/img/post/2022-07-13/kirchhoff-s-law2.jpg)_키르히호프의 전류 법칙, [출저](https://www.careerstoday.in/physics/kirchhoff-s-law)_

위 사진이 키르히호프의 전류 법칙을 잘 설명해주고 있습니다. 전류가 흐르는 회로에서 여러 전선이 겹치는 노드가 있을 때, 그 노드를 기준으로 노드에 들어가는 전류와 노드로부터 나가는 전류의 양이 같다는 것입니다. 

**키르히호프의 전압 법칙(Kirchhoff's Voltage Law)** :

![Kirchhoff's Voltage Law](/assets/img/post/2022-07-13/kirchhoff-s-law3.jpg)_키르히호프의 전압 법칙, [출저](https://www.careerstoday.in/physics/kirchhoff-s-law)_

키르히호프의 전압 법칙은 위 그림과 같이 회로의 형태가 닫혀있는(폐쇄형) 회로에서 적용됩니다. 키르히호프의 전압 법칙에 따르면 폐쇄형 회로에서 생기는 모든 전압을 합한 값은 0입니다.


## **커페시터(capacitor)**
케퍼시터는 부품 중 하나이며, 케퍼시터에 공급되는 전기를 자신의 역량만큼 저장하고 있다가 인가되는 전기가 중단되거나 줄었을 때 저장한 전기를 역으로 공급하는 기능을 합니다. 회로에서는 다음과 같이 나타내어집니다:

![capacitor symbol](/assets/img/post/2022-07-13/R.jpg){: width="500" height="300"}_케퍼시터 표기, [출저](https://www.stitchingworlds.net/experimentation/measuring-textile-capacitors/)_ | ![Capacitors](/assets/img/post/2022-07-13/Capacitors.jpg){: width="500" height="300"}_케퍼시터, [출저](https://en.wikipedia.org/wiki/Capacitor)_

케퍼시터의 원리를 알아내기 위해 당므과 같은 회로를 구상해보겠습니다(아래 왼쪽 그림2). 

![capacitor circuit](/assets/img/post/2022-07-13/capacitor_circuit.PNG)_capacitor circuit_ | ![charged capacitor circuit2](/assets/img/post/2022-07-13/capacitor_circuit2.PNG)_charged capacitor circuit_

회로에서는 전원부와 커페시터만이 연결되어 있고 전원부는 V만큼의 전위차를 만들어내며 시계 방향으로 전류를 공급하고 있습니다. 전류가 흐를 때 **전자는 전류의 반대방향으로 이동**하기 때문에, 그림에서 전자는 시계 반대방향으로 이동하게 됩니다. 하지만 그림에서 커페시터에 의해 회로가 끊어져 있기 때문에, 전자들은 커페시터에 막혀 오른쪽 넓적한 판에 갇혀있게 됩니다. 오른쪽 판에 전자가 더 많이 모이게 될 수록 오른쪽 판이 점점 음전하를 띄게 되며 반대쪽 왼쪽 판에 양전하가 모이게끔 하는 역할을 하게 됩니다. 이 과정이 지속되다가 커페시터의 역량이 한계에 도달하여 더이상의 전자 축적이 되지 않는 상태까지 온 경우, 위와 같은 상호작용이 멈추게 되며 더이상 회로에 전류가 흐르지 않게 됩니다. 

그럼 다음과 같은 상황에서 전원부를 탈착하면 어떤 현상이 일어날까요? 전원부가 없어지게 되면 커페시터가 저장된 전기를 배출하며 전원부 역할을 하게 됩니다. 대신 이번에는 왼쪽 판의 전위가 더 높기 떄문에(양전하) 반시계 방향의 전류를 생성해 내며, 그 말인 즉슨 오른쪽 판에 축적된 전자를 시계방향으로 배출한다는 의미가 됩니다.

![capacitor circuit without power](/assets/img/post/2022-07-13/capacitor_circuit3.PNG)_capacitor circuit without power_

커페시터에도 공식이 존재하는데, 그 공식은 다음과 같습니다:

$$q = cv$$

여기서 q는 전하량, v는 전압을 나타내고, c는 비례상수로써 커페시터의 용량을 나타냅니다. 커페시터가 얼마나 많은 전하를 저장할 수 있는지를 나타냅니다. 고로 c가 크면 클수록 커페시터가, 혹은 커페시터에 컬리는 전압이 크면 클수록 많은 전하를 저장할 수 있다는 의미가 됩니다. 

## **마치며**
이번 글에서는 SNN을 공부하기 전에 알아두면 유용한 기초적인 회로이론 내용을 다뤄보았습니다. 다음 글에서는 **SNN을 위한 회로이론** 시리즈의 최종목적지인 **RC회로가 무엇인지**, 그리고 **RC회로의 전압응답을 구하는 방법**에 대하여 알아보도록 하겠습니다.

글을 읽으시며 추가설명이 필요한 부분이나 이해가 어려운 부분, 혹은 오류가 있다면 댓글로 적어주시면 적극반영하도록 하겠습니다. 읽어주셔서 감사합니다.