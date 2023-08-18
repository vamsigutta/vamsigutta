---
title: "Building a Q&A Bot from Private Documents"
date: 2023-08-18T19:29:04+05:30
categories: ["chatbot", "ml", "QA"]
author: Vamsi Gutta
ShowToc: true
---

In today’s rapidly evolving digital landscape, having immediate access to accurate and reliable information is more crucial than ever. Whether you are a business owner aiming to improve customer service, a researcher needing quick answers, or just a tech enthusiast eager to dabble in artificial intelligence, building your Q&A bot using a private knowledge base could be the game-changer you are looking for.

Imagine a virtual assistant, tailored to your unique needs and preferences, that can sift through volumes of your curated data to deliver precise answers in an instant. It’s not just about convenience; it’s about maximizing efficiency and leveraging your data to empower decisions. This is where a Q&A bot shines — a tool that is no longer exclusive to tech giants but is within the reach of anyone with a bit of curiosity and determination.

In this blog post, we will embark on a journey to create your personalized Q&A bot, anchored securely on a private local directory. This means that your bot will be smart, but also respectful of data privacy, drawing answers from your own that you control and manage.

## In context learning

Before LLMs, the output of the model was dependent on the trained data. The LLMs these days possess the skill to learn and solve new tasks by providing input(prompt) without explicitly training them. This skill is called In context learning.

So in our method, we provide a context along with a query for the LLM model to get appropriate answers without hallucinations. Every LLM has a limited context window, i.e. we can only enter a limited number of tokens for context.

## Gathering the context

To gather the context for a given query, we create vector embeddings for all the document data. This data is stored in a vector database. We can use multiple machine learning methods like SVM and KNN to retrieve embeddings similar to the query.

In our example, we are considering a user directory that can have multiple files. We are considering only a few file types(pdf, txt, md, CSV) using langchain and are using the FAISS model to perform a similarity search on the embedding created using hugging face.

Faiss(Facebook AI Similarity Search) is a library for efficient similarity search and clustering of dense vectors. It contains algorithms that search in sets of vectors of any size. Faiss contains several methods for similarity search. It assumes that the instances are represented as vectors and are identified by an integer and that the vectors can be compared with L2 (Euclidean) distances or dot products. Vectors that are similar to a query vector are those that have the lowest L2 distance or the highest dot product with the query vector. It also supports cosine similarity, since this is a dot product on normalized vectors.

```
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS

def get_context_faiss(query):
    search_results = db.similarity_search(query)
    context_doc = ' '.join([ doc.page_content for doc in search_results])
    return context_doc

if __name__ == "__main__":
    documents = []

    # Mention here your custom path
    path = "./test/"
    for document in ["pdf", "md", "txt", "csv"]:
        loader = DirectoryLoader(path, glob=f'*.{document}')
        documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
    all_splits = text_splitter.split_documents(documents)

    embeddings = HuggingFaceInstructEmbeddings(
            model_name="hkunlp/instructor-large",
            model_kwargs={"device": "cpu"}
        )
    vectorstore = FAISS.from_documents(all_splits, embeddings)
    vectorstore.save_local("vector_store/faiss_index_local_data")
    db = FAISS.load_local("vector_store/faiss_index_local_data", embeddings)
    query = input("Enter your query: ")
    context = get_context_faiss(query)
```

## Prompting the LLM

Once the context is obtained, We create a prompt that is sent to the LLM model hosted locally/remotely. In our example, we are working on an LLM hosted locally. The following code shows how we can send our prompt to the LLM to get an answer.

```
from transformers import T5Tokenizer, T5ForConditionalGeneration
prompt_txt = f"""{context}
Q: {query}
A: 
"""       
## In the current example we are using Flan-T5-llm to get our answers
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl")

input_ids = tokenizer(prompt_txt, return_tensors="pt").input_ids

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))
```

## Enhancing Capability

1. The current code can be enhanced by including speech recognition, We have some open-source models which can help convert the speech to text offline without any credentials. The following code utilizes Openai's whisper to convert speech to text converter.

    ```
    import speech_recognition as sr
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something!")
        audio = r.listen(source)
    query = r.recognize_whisper(audio, language="english")
    ```
    So instead of typing a query, the user can directly talk to a machine to get the answers.

2. We can enhance the capability of the program by asking it to summarize the recorded calls or videos. The videos are converted to audio and further, this audio is converted to text which can be used for querying.

This post is for you to get started in building private bots which can enhance your productivity.

The complete code for this bot can be found [here](https://gist.github.com/vamsigutta/8505493931a7474b455d91729d164c18)

If you find my post interesting and feel like supporting me, do buy me [coffee](https://www.buymeacoffee.com/vamsigutta).

Happy coding!!