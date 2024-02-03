---
title: '[SNN Basic Tutorial 2] SNN을 위한 기초 뇌과학'
date: 2022-06-29 17:02:00 +0900
categories: ['[KOR] SNN Basic Tutorial']
tags: [SNN Basic Tutorial, SNN, Spiking Neural Network, Neuroscience, 뇌과학, 뉴런, 시넵스, 활동전위, action potential] # TAG names should always be lowercase
toc: true
---

### **SNN Basic Tutorial 목차**

1. [Spiking Neural Network란](https://jinprelude.github.io/posts/SNN-Basic-Tutorial-1-Spiking-Neural-Network%EB%9E%80/)

2. **[SNN을 위한 기초 뇌과학](https://jinprelude.github.io/posts/SNN-Basic-Tutorial-2-SNN%EC%9D%84-%EC%9C%84%ED%95%9C-%EA%B8%B0%EC%B4%88-%EB%87%8C%EA%B3%BC%ED%95%99/)**

3. [SNN을 위한 회로이론(1): 기초](https://jinprelude.github.io/posts/SNN-Basic-Tutorial-3-SNN%EC%9D%84-%EC%9C%84%ED%95%9C-%ED%9A%8C%EB%A1%9C%EC%9D%B4%EB%A1%A0(1)-%EA%B8%B0%EC%B4%88/)

4. [SNN을 위한 회로이론(2): RC회로](https://jinprelude.github.io/posts/SNN-Basic-Tutorial-4-SNN%EC%9D%84-%EC%9C%84%ED%95%9C-%ED%9A%8C%EB%A1%9C%EC%9D%B4%EB%A1%A0(2)-RC%ED%9A%8C%EB%A1%9C/)

5. [Leaky Integrate and Fire(LIF) 모델 설명](https://jinprelude.github.io/posts/SNN-Basic-Tutorial-5-Leaky-Integrate-and-Fire-%EB%AA%A8%EB%8D%B8-%EC%84%A4%EB%AA%85/)

딥러닝을 공부해 보신 분들은 아시겠지만 DNN 네트워크의 형상학적인 부분을 동물의 신경망에서 영감을 받았다는 점을 제외하고는 더이상 생물학과 관련된 지식을 요구하지 않습니다. 하지만 **[이전 포스트](https://jinprelude.github.io/posts/SNN-Basic-Tutorial-1-Spiking-Neural-Network%EB%9E%80//)** 에서 언급했듯이 SNN의 경우 태생이 생명과학에 뿌리를 두고 있는 만큼 대부분의 명칭이 생물학적 용어를 그대로 차용합니다. 본 글에서는 SNN의 기반이 되는 기초적인 뇌과학을 다룹니다. 이 글을 읽고 SNN을 보신다면 더욱 깊은 이해를 하실 수 있으실거라 믿습니다. 이번 글을 읽으시기 전에 **[이전 포스트](https://jinprelude.github.io/posts/SNN-Basic-Tutorial-1-Spiking-Neural-Network%EB%9E%80/)** 를 읽고 오신다면 이해가 더욱 수월하실 겁니다.

> 본 내용은 [Neuroscience: Science of the Brain](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjL__-ossL4AhVmGKYKHXu5BEUQFnoECAMQAQ&url=https%3A%2F%2Fbrain.mcmaster.ca%2FBrainBee%2FNeuroscience.Science.of.the.Brain.pdf&usg=AOvVaw1qxRUhwDpXU5UC86cs2wnK&cshid=1655946824320580)에 많은 기반을 두고 있습니다. 놔과학에 대한 지식이 전무한 제가 입문할 수 있도록 많은 도움을 준 자료입니다.
{: .prompt-info }

> 추후 SNN 공부에 쓰이게 될 용어에 있어서는 영어로 표기하겠습니다. 한영혼용이 읽으실 때 불편하실 수는 있어도 추후 관련 자료를 보실 때 더 도움이 되시리라 믿습니다.
{: .prompt-info }

## **Neuron의 구조**

DNN이 Neuron과 weight 두 가지 컴포넌트로 복잡한 연산을 처리하듯이, 동물의 뇌를 구성하는 가장 작은 단위는 **Neuron**입니다.

![Neuron의 명칭](/assets/img/post/2022-06-23/Neuron.jpg)_Neuron의 명칭, [출저](https://commons.wikimedia.org/wiki/File:Neuron.svg)_

저희는 뉴런에서 **Dendrite(수상돌기), Cell body(세포체), Axon(축삭돌기), Axon terminal(축삭돌기 말단)** 이 4가지 요소에 주목하겠습니다.

**Dendrite** 는 다른 뉴런들이 보내는 Spike를 받아 **cell body** 에 전달하게 됩니다. dendrite에는 **Dendritic spine(수상돌기가시)** 이라는 돌기 가시가 있는데(아래 좌측 그림 참고), dendritic spine마다 하나의 뉴런이, 정확히는 뉴런의 **Axon terminal**이 연결되어 있습니다. 즉, 한 줄기의 dendrite에 한 개 이상의 뉴런이 연결관계에 있습니다. 이렇게 dendrite로부터 종합된 정보들은 cell body에서 다른 dendrites들의 정보와 함께 다시 종합됩니다. 종합된 spike로 인한 action potential(활동전위)이 threshold voltage를 넘게 되면 **Axon** 을 통해 action potential을 전달하고, 그 연쇄반응으로 synapse에서는 또다른 뉴런의 dendrit
ic spine을 통해 신호를 전달하기 위한 화학반응을 일으키게 됩니다. 아래 우측 그림은 기능적인 요소에 기반하여 뉴런을 단순화한 그림입니다.

![dendritic spine](/assets/img/post/2022-06-23/Dendritic-Spine0.jpg)_dendritic spine, [출저](https://www.msrblog.com/science/biology/dendritic-spine.html)_ | ![Simplified Neuron](/assets/img/post/2022-06-23/neuron_simplified.PNG){: w="700" h="400" }_Simplified Neuron, [출저](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjL__-ossL4AhVmGKYKHXu5BEUQFnoECAMQAQ&url=https%3A%2F%2Fbrain.mcmaster.ca%2FBrainBee%2FNeuroscience.Science.of.the.Brain.pdf&usg=AOvVaw1qxRUhwDpXU5UC86cs2wnK&cshid=1655946824320580)_

## **Action Potential이란**

뇌에서는 2가지 신호전달 방법이 있습니다. 하나는 **화학적 신호**, 하나는 **전기적 신호**입니다. 뉴런과 뉴런간에 신호가 전달될 때, 즉, dendritic spine과 axon terminal 간에는 화학적 신호전달이 일어나고, action potential 이 threshold를 넘어서 cell body에서 spike가 생성되어 axon을 통해 axon terminal로 spike를 보낼 때는 전기적 신호전달이 일어납니다. 먼저 뉴런 내부에서 전기적 신호전달이 어떻게 일어나는지, 즉, Action potential에 대해 알아보겠습니다.

Action potential을 알기 위해서는 먼저 Membrane potential이 무엇인지 알아야 합니다. 뉴런은 **Membrane(세포막)**으로 감싸져 있으며 세포막을 기준으로 내부와 외부가 나뉘고, 내부와 외부 사이에 전위차가 존재합니다. 뉴런의 내부는 외부와 비교해 -65mv만큼의 전위차를 가지는데, 이 전위차를 **Membrane Potential(막전위)** 이라고 합니다. 그리고 뉴런의 내외부에는 양전하를 띄는 **Sodium(나트륨)** 과 **Potassium(칼륨)** 이 존재하는데, 내부에는 Potassium이, 외부에는 Sodium이 더 많이 분포해 있습니다(아래 그림 참조).

![membrane potential](/assets/img/post/2022-06-23/membrane_potential.jpg)_membrane potential, [출저](https://jackwestin.com/resources/mcat-content/plasma-membrane/membrane-potential)_

그럼 Action potential이란 무엇일까요? 위키피디아에 따르면 "근육·신경 등 흥분성 세포의 흥분에 따른 막 전위의 일시적 변화"라고 설명합니다([출저](https://ko.wikipedia.org/wiki/%ED%99%9C%EB%8F%99%EC%A0%84%EC%9C%84)). 저희가 배우는 목적에 맞게 가공해보면 **뉴런의 흥분(excitation)에 따른 Membrane potential의 일시적 변화**라고 할 수 있겠습니다. 정의를 잘 기억하며 더 자세히 알아보겠습니다.

Action potential은 cell body가 dendritic spines들로부터 다른 뉴런들의 화학적 신호를 받으면서 만들어집니다(화학적 반응에 대해서도 다음 절에서 알아보도록 하겠습니다). 특정 Dendritic spine에서 화학적 신호 전달이 이루어지게 되면 membrane에 붙어있는 **ligand-gated Sodium channel(리간드 개폐 통로)** 가 반응하여 열리게(활성화하게) 됩니다.

Ligand-gated Sodium channel의 이름을 분석해보며 역할을 알아보겠습니다. **channel(체널)** 이란 세포막(membrane)에 붙어서 출입통로의 역할을 하는 기관으로 channel의 종류에 따라 다양한 조건에서 다양한 물질들을 출입시킵니다. 위 사진에서 노란색, 파란색, 보라색으로 그려진 것들이 세포막에 붙어있는 체널을 그린 것입니다. Sodium은 말 그대로 sodium을 드나들도록 한다는 뜻이며 Ligand-gated의 의미는 Neurotransmitter(신경전달물질)에 의해 개폐가 이루어진다는 의미를 가지고 있습니다. Neurotransmitter과의 상호작용에 대해서는 화학적 반응 챕터에서 더 자세히 알아보도록 하겠습니다.

Ligand-gated Sodium channels들이 활성화되면 channel을 통해 외부의 sodium이 유입되며, 이로 인해 -65mV의 전위차를 가지고 있던 membrane potential의 값이 상승하게 됩니다. 하지만 뉴런의 membrane은 전류를 완전히 차폐하지는 못하는데, 때문에 만약 membrane potential의 값이 상승폭이 충분히 크지 못하면 membrane을 통한 전위 누수(Leak)가 일어나는 크기가 더 크게 되고, membrane potential은 다시 -65mV로 되돌아가게 됩니다. 반면 충분한 신호 전달이 이루어져 Membrane potential이 -55mV가 넘어가게 되면, 이때부터는 **voltage-gated sodium channels(위 그림에서 보라색 채널)**들이 열리게 됩니다. 이름에서 유추해볼 수 있듯이 이 체널은 membrane potential에 반응하며 -55mV가 되면 활성화되어 외부의 sodium을 뉴런 내부로 유입하게 되고, 이로 인해 membrane potential의 값이 -55mV에서 30~40mV까지 치솟게 됩니다. 고로 action potential은 membrane potential이 -55mV만 넘게 되면 시작된다고 볼 수 있고, 이 값을 threshold voltage라고 합니다.

![action potential gif](/assets/img/post/2022-06-23/actionp1.gif)_action potential의 visualization, [출저](https://faculty.washington.edu/chudler/ap.html)_

Membrane potential이 30~40mV에 다다르게 되면 열린 voltage-gated sodium channel들은 닫히게 되고(ligand-gated Sodium channel은 neurotransmitter이 떨어지면서 먼저 닫히게 됩니다) **voltage-gated potassium channel(위 그림에서 노란색 채널)**이 열리게 되는데, 이 potassium channel은 내부의 potassium을 외부로 배출시키는 역할을 합니다. 더이상의 sodium 유입이 막히고 potassium이 배출되기 시작하며 membrane potential은 급격히 떨어지기 시작합니다. 그리고 막전위가 -65mV 이하로 떨어지게 되면 potassium channel 또한 닫히게 되며 action potential이 끝나게 됩니다. 이 때 potassium의 배출에 관성이 붙게 되며 초기 membrane potential인 -65mV보다 더 떨어지게 되는데, 이는 **sodium-potassium pump(위 그림에서 하늘색 채널)** 에 의해 서서히 조정되어 다시끔 -65mV로 원상복구 되어집니다.

sodium-potassium pump는 뉴런 내부의 sodium 3개와 외부의 potassium 2개를 맞바꾸는 역할을 수행하는 channel입니다. 이 sodium-potassium pump는 membrane potential을 원상복구해주는 것에 더해서 뉴런의 excitation으로 인해 급격하게 늘어난 뉴런 내부의 sodium, 그리고 급격하게 줄어든 potassium 또한 원상복구해 주는 역할을 수행합니다. 이 pump는 neurotransmitter 혹은 voltage에 의해 자동으로 작동하는 channels들과는 달리 별도의 에너지인 ATP가 있어야 작동하며, 때문에 뉴런 내부의 미토콘드리아에 의해 생성된 ATP를 소모하며 작동합니다. 

아래 그림은 action potential의 발생 단계를 직관적으로 시각화하였습니다.

![action potential의 단계](/assets/img/post/2022-06-23/ap3.gif)_action potential의 단계, [출저](https://faculty.washington.edu/chudler/ap.html)_

## **화학적 신호전달: Neurotransmitter와 Synapse**

위와 같은 과정을 통해 만들어진 action potential은 axon을 따라 axon terminal로 이동합니다. 아래 그림에서 알 수 있듯이 action potential이 만들어진 뉴런의 axon terminal과 연결되어있는 neuron의 dentritic spine 사이에는 공간이 있기때문에 전기적 신호가 그대로 전해질 수는 없습니다. 때문에 Neuron과 Neuron끼리 신호를 전달할 때는 화학적 신호전달을 사용합니다. Action potential이 일어난 뉴런의 axon terminal과 연결되어있는 neuron의 dendritic spine을 포함한 공간을 **Synapse(시냅스)** 라고 칭합니다. 그리고 그 사이에 있는 공간을 **Synaptic cleft**라고 합니다. 

![neuron and synapse](/assets/img/post/2022-06-23/Biological-neuron-and-synapse.png)_neuron and synapse, [출저](https://www.researchgate.net/figure/Biological-neuron-and-synapse_fig1_324229756)_

Synapse를 기준으로 정보를 전달해주는(action potential이 생성된) 뉴런을 시냅스 뒤에 있다는 의미의 **Pre-synaptic Neuron**, 정보를 받는 뉴런을 **Post-synaptic Neuron**이라고 칭합니다. 이제 action potential이 생성된, 즉, excitation이 일어난 pre-synaptic neuron이 어떻게 post-synaptic neuron에게 화학적으로 신호를 전달해주는지 알아보도록 하겠습니다. 

아래 그림은 synapse 부분을 확대한 그림입니다. Axon terminal , dendritic spine, synaptic cleft 가 모두 잘 보입니다. Axon terminal에는 synaptic vesicles들이 존재하고 있습니다. 이 synaptic vesicle 안에는 위에서도 언급된 neurotransmitter들이 들어있습니다. 또한 Axon terminal의 세포막에는 **Voltage-gated Ca channel**이 붙어있는데(이제는 이름을 통해 어떤 역할인지 유추하실 수 있으실 겁니다), action potential이 도착해 membrane potential이 높아짐에 따라 channel이 열리게 되면 외부에서 칼슘이온이 axon terminal 내로 유입되게 됩니다. 칼슘이 들어오게 되면 synaptic vesicles는 synaptic cleft와 가까운 쪽의 세포막과 결합하게 되며, 이에 따라 synaptic vesicles 안에 있던 neurotransmitter가 synaptic cleft에 배출되게 됩니다. 배출된 neurotransmitter들은 post-synaptic neuron에 특정 채널과 결합하게 되는데, 이 채널이 바로 위에서 언급했던 Ligand-gated Sodium channel입니다. Ligand-gated Sodium channel은 neurotransmitter와 결합됨에 따라 활성화되며, 외부에 있는 sodium들이 Ligand-gated Sodium channel을 통해 post-synaptic neuron에 유입되게 됩니다. 이는 post-synaptic neuron의 membrane potential을 높이게 되고, 이 다음부터는 위에서 살펴본 action potential의 단계가 post-synaptic neuron에서 다시 시작되게 됩니다.

![synapse](/assets/img/post/2022-06-23/Synapse-Diagram.jpg)_synapse, [출저](https://www.sciencefacts.net/synapse.html)_

Neurotransmitter에는 Glutamate, Glycine, GABA 등 다양한 종류가 있으며, 그에 따라 Ligand-gated channel도 sodium에 반응하는 채널 외에도 다양한 종류가 있습니다. 본 글에서는 neurotransmitter과의 상호작용에 의해 post-synaptic neuron의 전위차를 상승시켜주는 경우만 살펴보았지만, 특정 ligand-gated ion channel은 전위차를 되려 낮추기도 합니다. 본 글의 목적은 SNN을 이해하기 위해 뇌과학의 기반을 다지기 위함으로 여기서 설명을 마무리하겠습니다.


## 마치며
이번 글에서는 SNN의 근간이 된 동물의 실제 뉴런과 시냅스의 구조와 역할에 대하여 알아보았습니다. 다음 글에서는 SNN을 이해하는 데에 필요한 **기초 회로이론**에 대하여 알아보겠습니다.

글을 읽으시며 추가설명이 필요한 부분이나 이해가 어려운 부분, 혹은 오류가 있다면 댓글로 적어주시면 적극반영하도록 하겠습니다. 읽어주셔서 감사합니다.
