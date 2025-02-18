---
title: "Reflections on the Success Factors of LLMs and the Future of AI"
date: 2025-02-18 10:26:00 +0900
categories: ['Reflections']
tags: [LLM, World-Model, Continual-Learning, Meta-Learning, Robot-Learning]
author: euijin_jeong
toc: true
---

## Reflection on the Success Factors of LLMs  

I believe the key success factor of Large Language Models (LLMs) lies in **world-model learning**. World-model learning refers to the ability to predict and understand the dynamics of an environment. LLMs acquire such a world model in an unsupervised manner through next-token prediction on vast amounts of text data (task-unlabeled data). This world model underpins the impressive capabilities of LLMs, including few-shot learning, in-context learning, and planning.  

My conviction in this perspective solidified after the release of OpenAI’s o1. Unlike previous models, o1 can improve its responses through **thinking**. Some argue that this is due to reinforcement learning, but if o1 were merely producing answers via reinforcement learning without the ability to think over time, it wouldn’t have been as successful. I believe that **thinking** is essentially **world-model searching**. Since an LLM itself serves as a well-defined world model, it performs **planning and search** within this model to refine its responses.  

---

## Beyond World-Model Learning  

World-model learning remains an unsolved challenge. We still struggle with applying it to multi-modal data. However, LLMs have demonstrated the power of next-token prediction, and this will eventually lead us to a better understanding of world models for multi-modal domains. But what comes next?  

In his talk, ["Sequence to sequence learning with neural networks: what a decade"](https://www.youtube.com/watch?v=1yvBqasHLZs), Ilya Sutskever presents four keywords in his final slide that he believes are crucial for achieving superintelligence:  

1. **Agentic**  
2. **Reasons**  
3. **Understands**  
4. **Is self-aware**  

However, these terms are quite surface-level. Earlier in the talk, Ilya states that **the era of representation learning is over**. He argues that while computing power continues to increase, data is not keeping pace. He likens data to fossil fuels, claiming that we are already facing a **data shortage**, necessitating algorithmic innovations.  

So, what could address this issue?  

My personal answer is **continual learning**. The ultimate goal of continual learning is to enable models to learn from **streaming data**, much like humans do. Continual learning involves **efficiently balancing forgetting and memory within a limited network**, and in the process, new capabilities can emerge. **Meta-learning**, which accelerates future learning based on prior knowledge (learning to learn), is likely to arise naturally from continual learning—provided that **sufficient representation learning is embedded within it**.  

If a continual learning agent is capable of representation learning, it would be able to accumulate vast amounts of knowledge simply by existing in the world.  

---

## The Current State of Robot Learning  

I believe the field of **robot learning** has yet to fully leverage the world-model learning approach demonstrated by LLMs. Current models such as **pi0 and OpenVLA** primarily focus on **action prediction**, which is an extension of **behavior cloning (BC)** into a more generalized form. This approach remains **task-labeled**, relying on instruction-following datasets. However, it fails to exploit **unsupervised learning from large-scale unlabeled data**, a capability that has been crucial for LLMs.  

To overcome this limitation, I propose introducing **next observation prediction** into robot learning. This involves training robots to predict the next observation based on their current observation and action.  

| Category       | Language Models               | Robot Learning Models         | Characteristics                                                      |
| -------------- | --------------------------- | ----------------------------- | -------------------------------------------------------------------- |
| Task-labeled   | QA datasets                 | Instruction datasets (DROID, OXE) | Used for downstream tasks, focuses on instruction following         |
| Task-unlabeled | Web-crawled data            | Unlabeled robot interaction data | Enables skill learning and world-model learning                      |
| Random data    | Randomly generated sentences | Transitions from a random policy | Not useful for language models but useful for learning world dynamics in robots |

Currently, robot learning primarily depends on **task-labeled datasets** (e.g., DROID, OXE, RT-1, RT-2). Since action prediction models can only learn effectively from such labeled data, they lack the ability to acquire general skills from **task-unlabeled** data.  

However, with **next observation prediction**, robots could also learn useful skills from **task-unlabeled data**. Even though these datasets lack explicit task labels, they still contain **intentional behaviors aimed at achieving specific objectives**—such as object manipulation, exploration, and repetitive patterns. By learning to predict future observations, a robot can implicitly acquire **general skills** and develop representations that can be applied to a variety of downstream tasks.  

Unlike in language models, where random data is not particularly useful, random interaction data in robots may still be valuable for learning dynamics. Research on the utility of random data in robot learning is planned, but for now, the primary focus is on leveraging task-unlabeled data. While it remains to be fully validated, incorporating next observation prediction may allow robots to develop more generalizable skills that improve their performance on downstream tasks, including instruction following.