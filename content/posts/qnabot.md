---
title: "Build your own QA Chatbot in Python"
date: 2023-07-23T19:45:42+05:30
categories: ["chatbot", "ml", "QA"]
author: Vamsi Gutta
ShowToc: true
---
## Introduction 

Hello everyone! With the advent of machine learning models and the recent introduction to large language models, interaction with a chatbot has become a part of our lives. Every product we build has some documentation as well as FAQs. Building a chatbot that can answer the FAQs would reduce the burden of searching through the FAQs and the load for the support staff. In this blog, we are going to build one such bot.

## Problem statement

Every machine-learning problem can be solved in multiple ways. Each solution depends on how you approach the problem. For example, a chatbot can be built using one of the following methods.

1. **Classification Problem**: In this method, we arrange a question and its variations under a tag along with its answer. Given a query, our model should classify which tag it belongs to give an appropriate answer.

2. **Cosine Similarity and Embedding Vectors**: In this method, given a query, we need to identify which question the given query is similar to and return the answer mapped to it. Adding vector embeddings helps in finding semantic similarity.

3. **Siamese Network**: This method is similar to finding the similarity between the questions, but instead of depending on the cosine similarity, we use convolutional neural networks to identify the similarity.

4. **Sequence-to-Sequence Problem**: In this method, we predict the output sequence (answer) based on the input sequence(query). We use an encoder-decoder architecture to solve this problem.

5. **Knowledge-graph and NLP**: In this method, a knowledge base needs to be built on the answers. We have to use NLP techniques to identify the intent of the questions and map it to the correct answer.

6. **Reinforcement learning**: In this method, An agent(bot) views the dialogue as a sequence of states and actions, intending to maximize the reward. The bot learns by observing a dialogue sequence(question & answer) and then tries to replicate it.

The above-mentioned methods is not an extensive list, but a subset of the solutions.

## Building the bot

In this blog, we are building a bot that identifies the cosine similarity from the given query as this is one of the easiest ways to build a bot.

### Step 1: Data preparation

We need pre-determined questions & answers mapped as shown below.
```python
questions = ["hi", "What is the product", "what is the cost of the product"]
answers = ["Hi there, How can I help you", "The product is a QA chatbot", "The product is available for free"]
```

### Step 2: Create vector embeddings for the questions

There are multiple ways to create vector embeddings. In this blog, we will be using the pre-trained models from sentence transformers.

```python
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
model.max_seq_length = 512
embeddings = model.encode(
        questions,
        show_progress_bar=True,
        convert_to_tensor=True
    )
```

### Step 3: Determine the cosine similarity

In this step, we compute the embedding of the query and compare it with the existing embeddings to understand which question is similar. The question with maximum cosine similarity is selected. The corresponding answer mapped to the question is the reply.

```python
query_embeddings = model.encode(query, convert_to_tensor=True)
cosine_scores = util.cos_sim(query_embeddings, embeddings)
similarity_score = torch.max(cosine_scores)
answer_idx = torch.argmax(cosine_scores)
answer = answers[answer_idx]
```

Voila! We've built a chatbot.


## Improving the solution

Building a chatbot with vector embeddings can help in providing answers to multiple variations of questions. To answer more variations, we need to make a sufficient number of variations to help it understand the similarity. 

This post is for you to get started and think of different ways to build a bot for a task using machine learning.

The complete code for this bot can be found [here](https://gist.github.com/vamsigutta/e12747aa035c8d0fccb037cfd1fc3eea)

If you find my post interesting and feel like supporting me, do buy me [coffee](https://www.buymeacoffee.com/vamsigutta).

Happy coding!!
