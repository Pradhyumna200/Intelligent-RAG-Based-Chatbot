\# Intelligent RAG-Based Chatbot Using LangChain and LLMs



\## Project Overview

This project implements a Retrieval-Augmented Generation (RAG) chatbot for customer-support questions. It retrieves relevant text from a customer-support dataset stored in ChromaDB and uses a language model to generate context-based answers.



\## Problem Domain Description



Customer support teams receive a large number of customer queries related to service issues, complaints, account assistance, and support channels. Finding relevant information from large collections of customer-support conversations can be time-consuming. Traditional chatbots may also provide generic answers that do not address the user's specific question.



This project develops an Intelligent Retrieval-Augmented Generation (RAG) Chatbot to answer customer-support questions using information retrieved from a knowledge base. The chosen knowledge base is the Twitter Customer Support dataset (twcs.csv), which contains customer-support conversations.



The project processes the first 10,000 rows of the dataset, producing 9,980 unique non-empty text records. The text is divided into overlapping chunks of 500 characters with an overlap of 50 characters. Sentence Transformer embeddings are generated and stored in a persistent ChromaDB vector store.



When a user submits a question, the system retrieves relevant text from the vector store and provides the retrieved context to a language model to generate a response. The chatbot interface is built using Streamlit. RAGAS metrics, including Faithfulness and Answer Relevancy, are used to evaluate the generated answers.



The main goal is to provide context-based responses to customer-support questions while reducing unsupported answers. The current implementation uses a subset of the dataset, and the retrieval quality and answer relevance can be improved through further experimentation.





\## Objectives

\- Build a RAG pipeline for customer-support queries.

\- Process and split text into manageable chunks.

\- Generate embeddings and store them in ChromaDB.

\- Retrieve relevant context for user questions.

\- Generate answers using a language model.

\- Evaluate responses using Faithfulness and Answer Relevancy metrics.



\## Technologies Used

\- Python

\- LangChain

\- ChromaDB

\- Hugging Face Sentence Transformers

\- Groq API for RAGAS evaluation

\- Google Gemini for the configured chatbot pipeline

\- Streamlit

\- Pandas, NumPy, and Jupyter Notebook

\- RAGAS



\## Project Structure



```text

Intelligent\_RAG\_Based\_Chatbot/

├── app.py

├── Intelligent\_RAG\_Based\_Chatbot.ipynb

├── README.md

├── ragas\_evaluation\_results.csv

├── sample\_questions\_answers.csv

├── .gitignore

└── vector\_db/

```



\## Dataset Source and Preprocessing



The dataset used in this project is Customer Support on Twitter, obtained from Kaggle.



\- Dataset Name: Customer Support on Twitter

\- Source: \[Kaggle – Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)

\- Dataset File: twcs.csv

\- Original Dataset Size: Over 3 million tweets and replies.



Due to the large size of the original dataset (approximately 5 GB), the first 10,000 rows were selected for this project. After preprocessing and removing duplicate and empty text records, 9,980 unique, non-empty text records were retained for building the RAG knowledge base.



The text was split into chunks of 500 characters with an overlap of 50 characters. Sentence Transformer embeddings were generated and stored in a persistent ChromaDB collection named customer\_support.



\### Dataset Setup



The original dataset is not included in the project submission because of its large size.



To reproduce the preprocessing pipeline:



1\. Visit the \[Kaggle dataset page](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter).

2\. Download the dataset and locate twcs.csv.

3\. Place the file in the following directory:



&#x20;  data/archive/twcs/twcs.csv



The project also includes a persistent ChromaDB vector store containing the processed embeddings, allowing the chatbot to reuse the existing knowledge base without rebuilding the index.



\## System Architecture



```text

Customer Support Dataset (twcs.csv)

&#x20;             |

&#x20;             v

&#x20;    Data Loading \& Cleaning

&#x20;             |

&#x20;             v

&#x20;       Text Chunking

&#x20;  (500 characters, 50 overlap)

&#x20;             |

&#x20;             v

&#x20; Sentence Transformer Embeddings

&#x20;             |

&#x20;             v

&#x20;  Persistent ChromaDB Vector Store

&#x20;             |

&#x20;             v

&#x20;       User's Question

&#x20;             |

&#x20;             v

&#x20;      Similarity Retrieval

&#x20;             |

&#x20;             v

&#x20; Retrieved Context + User Query

&#x20;             |

&#x20;             v

&#x20;   Language Model (Gemini)

&#x20;             |

&#x20;             v

&#x20;      Generated Answer

&#x20;             |

&#x20;             v

&#x20;     Streamlit Chatbot

```

\## RAG Pipeline



1\. Load and preprocess customer-support text.

2\. Split text into chunks.

3\. Generate text embeddings.

4\. Store embeddings in ChromaDB.

5\. Retrieve relevant context for a user's question.

6\. Generate an answer based on the retrieved context.

7\. Evaluate answer Faithfulness and Answer Relevancy.



\## Running the Streamlit Application



1\. Install the required dependencies in your Python environment.

2\. Configure the API keys in the .env file.

3\. Open Command Prompt in the project directory.

4\. Run:



```bash

python -m streamlit run app.py --server.fileWatcherType none

```



5\. Open the local URL displayed in the terminal, usually http://localhost:8501.



\## Sample Customer-Support Questions and Answers



The following 10 examples were prepared for RAGAS evaluation using retrieved customer-support contexts. Contact details and instructions are based on the dataset and have not been independently verified as current.



1\. Q: How can I contact customer support?  

&#x20;  A: Call customer support at 1-866-216-1072 or 888-269-9693.



2\. Q: How can I get an update on my complaint?  

&#x20;  A: You can expect an update from Oliver, who said they will look into your complaint and contact you with the next steps.



3\. Q: What should I do if I have a service problem?  

&#x20;  A: If you’re experiencing a service problem, contact Comcast support and provide your account number and full service address. If the issue continues, follow up with them and mention any repeated outages or area-wide problems.



4\. Q: Can I contact support through direct messages?  

&#x20;  A: Yes, you can reach support by sending a direct message for assistance.



5\. Q: How can I follow up on an unresolved complaint?  

&#x20;  A: The support team stated that they would look into the complaint and contact the customer with an update. The retrieved context does not provide the actual email address or telephone number, so those contact details cannot be confirmed from the available information.



6\. Q: What information should I provide to customer support?  

&#x20;  A: The retrieved context does not provide enough details to confirm exactly what information you should provide to customer support.



7\. Q: How can I request help with my account?  

&#x20;  A: The prepared answer refers to a link in the retrieved context. Verify that link before treating it as an official support channel.



8\. Q: How do I report a problem with a service?  

&#x20;  A: To report a problem with your service, send a direct message that includes your account number and full service address. If it’s an outage, you can also contact @comcastcares.



9\. Q: How can I ask the support team to investigate an issue?  

&#x20;  A: You can ask the support team to investigate by calling 1-866-216-1072, sending the issue details through the provided link, and submitting official feedback through the eForm.



10\. Q: What should I do if my issue has not been resolved?  

&#x20;   A: If your issue still hasn’t been resolved, contact customer support again and provide any additional details they request. You can also send a direct message with more information so they can investigate further and get back to you with a resolution.



\## Evaluation



The RAGAS evaluation was completed on 10 question-answer examples using Faithfulness and Answer Relevancy metrics.



| Metric | Average Score |

|---|---:|

| Faithfulness | 0.8000 |

| Answer Relevancy | 0.6665 |



Faithfulness measures whether generated answers are supported by the retrieved contexts. Answer Relevancy measures how relevant the generated answers are to the original questions.



These results provide an initial assessment of the chatbot's answer grounding and relevance across the 10 evaluated examples. The scores indicate opportunities to improve retrieval quality and answer relevance.



The detailed evaluation results are saved in ragas\_evaluation\_results.csv, and the 10 sample question-answer pairs are saved in sample\_questions\_answers.csv.



\## Limitations

\- The current vector store was built from a subset of the full dataset.

\- Retrieval quality and answer relevance require further improvement.

\- Generated answers depend on the quality of retrieved context and the configured language model.

\- The sample Q\&A pairs are prepared evaluation examples based on retrieved customer-support contexts; contact details and instructions have not been independently verified as current.



\## Future Enhancements

\- Process more of the available dataset.

\- Improve chunking and retrieval strategies.

\- Improve answer relevance and grounded response quality.

\- Expand the evaluation dataset. 

\- Add stronger error handling and deployment support.



\## Security

Keep API keys in the local .env file. Do not commit API keys, private credentials, or unnecessary large datasets to a public repository.



