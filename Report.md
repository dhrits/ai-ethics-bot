# 0. The Problem
While organizations would like to invest in the latest and greatest AI technologies and build safe products with good ROI, AI systems tend to cause anxiety born of a lack of understanding. In particular, people are concerned about the impact of AI on society. Additionally, nobody seems to understand the right way to approach building ethical and useful AI applications which will be a net benefit to humanity. 

People also seem anxious about the Government and the Administration's response to AI systems. Especially as the thinking around large AI systems is evolving quickly. 

# 1. Dealing with the Data
To alleviate some of the concerns outlined above, we've identified two documents: 

1. 2022: [Blueprint for an AI Bill of Rights](https://www.whitehouse.gov/wp-content/uploads/2022/10/Blueprint-for-an-AI-Bill-of-Rights.pdf) Making automated systems work for the American People - A document published by the White House Office of Science and Technology Policy in October 2022. This document outlines five key principles which should be followed with a goal of building safe AI Systems which serve the public rather than threaten rights and opportunities. 

2. 2024: [National Institute of Standards and Technology (NIST) Artificial Intelligent Risk Management Framework](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf) - A document published by NIST which aims to provide guardrail suggestions against the development of risky AI systems. 

To alleviate these concerns, we begin by indexing these documents in a manner that the most relevant context can be retrieved given a question from a stakeholder. 

Given stakeholders querying this system are unlikely to know too many details about these reports, a good **default chunking strategy** can be based on `RecursiveCharacterTextSplitter` using the latest OpenAI embedding model. `text-embedding-3-small` serves as a good tradeoff between computational complexity and performance. This will be the default choice. 

**Another chunking strategy** which I would like to try is the experimental `SemanticChunker`from OpenAI which aims to chunk sentences based on their semantic properties. Coupling this with `text-embedding-3-large` may offer even better performance at a tradeoff of higher computation and cost 

**This data indexing is done at the notebook linked [here](https://github.com/dhrits/ai-ethics-bot/blob/main/nbs/Dealing%20with%20Data.ipynb).**

# 2. An End-to-End Chatbot
An end-to-end prototype of a chatbot which allows querying the docs above is **deployed using a huggingface space [here](https://huggingface.co/spaces/deman539/ai-ethics-bot)**

The code for this application can be found in this [huggingface repository](https://huggingface.co/spaces/deman539/ai-ethics-bot/tree/main). 

The tech stack makes use of Chainlit to deploy the application and Langchain to build a simple RAG pipeline. Chainlit helps very quickly and cleanly build prototype systems based on chatbots and lanchain. This made it the obvious deployment choice. 

Langchain similarly allows for expressive LLM applications by making use of the concept of [chains](https://python.langchain.com/v0.1/docs/modules/chains/). In the end, we went with a simple RAG Chain which felt most expressive and reactive. **Experiments with a few chains can be found at the notebook [here.](https://github.com/dhrits/ai-ethics-bot/blob/main/nbs/Prototype%20Chain.ipynb)**

# 3. Creating a Golden Test Dataset
Following the principles of MDD, we would like to create a dataset which can be used to evaluate different strategies. We do this using the RAGAS framework. **[This notebook](https://github.com/dhrits/ai-ethics-bot/blob/main/nbs/Test%20Data%20and%20RAGAS%20Evaluation.ipynb)** contains the code for generating the golden **[test dataset](https://github.com/dhrits/ai-ethics-bot/blob/main/nbs/golden_eval_set.csv)**.

We then proceed to evaluate our two chunking strategies using the RAGAS framework on the metrics of faithfullness, answer-relevance, context-recall, context-precision and answer-correctness. **This evaluation and a table of comparison is available in the [same notebook](https://github.com/dhrits/ai-ethics-bot/blob/main/nbs/Test%20Data%20and%20RAGAS%20Evaluation.ipynb)** (Table of comparison at the end of the notebook).

**Evaluation Conclusions** - Both strategies perform well out-of-the-box. All metrics were > 80% except answer-correctness which was in the 60s. Overall, using SemanticChunker with text-embedding-3-large didn't make too much of a difference in the metrics. 

# 4. Fine-Tuning Open Source Embeddings
To improve the retrieval performance of our chatbot, **we finetune a set of open-source embeddings in [this notebook](https://github.com/dhrits/ai-ethics-bot/blob/main/nbs/Fine_Tuning_nomic_embed_text_v1_on_AI_Ethics_Docs.ipynb)**. This finetunes nomic-ai/nomic-embed-text-v1 model on AI Ethical Framework documents linked above.

This model is ranked 8 on the MTEB leaderboard for models < 250M parameters. The hope was that this model, once finetuned on bespoke data, will outperform off-the-shelf models.

# 5. Assessing Performance
We then use the assess the performance of the finetuned embeddings using RAGAS and our earlier generated [golden dataset](https://github.com/dhrits/ai-ethics-bot/blob/main/nbs/golden_eval_set.csv) in the [same notebook](https://github.com/dhrits/ai-ethics-bot/blob/main/nbs/Fine_Tuning_nomic_embed_text_v1_on_AI_Ethics_Docs.ipynb).

We compare the performance our our earlier chunking strategies with  chunking strategy based on the finetuned model. **[A table of comparison is linked at the end of the notebook](https://github.com/dhrits/ai-ethics-bot/blob/main/nbs/Fine_Tuning_nomic_embed_text_v1_on_AI_Ethics_Docs.ipynb)**. While there is a small change in most metrics, **answer-correctness goes up by 5%**.

**Since correctness is of paramount importance when building a system like this (to ensure trust with stakeholders), this is the best model to deploy.**

# 6. Managing Expectations and the Story
Modern AI-based systems have the potential to impact society in a great many ways. Like any disruptive technology, they can do good or harm depending on how they are deployed. With this in mind, several instituitions including The White House Office of Science and Technology Policy and National Institute of Standards (NIST) have come up with frameworks on how to manage AI Risks. 

To help stakeholders explore these documents, we have deployed a chatbot which can answer broad questions using these documents as a guide. This chatbot is available at at [this link](https://huggingface.co/spaces/deman539/ai-ethics-bot). 

Based on a summary provided by this bot, AI Systems can be safely deployed by following a few guidelines:

1. Regular Evaluation for Safety Risks:
    AI systems should be regularly evaluated for safety risks, ensuring that the residual negative risk does not exceed the risk tolerance and that the system can fail safely, especially when operating beyond its knowledge limits. Safety metrics should reflect system reliability, robustness, real-time monitoring, and response times for AI system failures.
2. Assessment of Adverse Impacts:

    Assess adverse impacts, including health and wellbeing impacts for value chain or other AI actors exposed to sexually explicit, offensive, or violent information during AI training and maintenance.
3. Assessment of Harmful Content and Bias:

   Evaluate the existence or levels of harmful bias, intellectual property infringement, data privacy violations, obscenity, extremism, violence, or CBRN (Chemical, Biological, Radiological, and Nuclear) information in system training data.
4. Re-evaluation of Safety Features:

    Re-evaluate safety features of fine-tuned models when the negative risk exceeds organizational risk tolerance.
5. Review of AI System Outputs:

   Review AI system outputs for validity and safety, including generated code to assess risks that may arise from unreliable downstream decision-making.
6. Implementation of Ethical Principles and Frameworks:

    Adoption of ethical principles and frameworks by organizations, such as the Department of Energy's AI Advancement Council, the Department of Defense's Artificial Intelligence Ethical Principles, and the U.S. Intelligence Community's Principles of Artificial Intelligence Ethics.
7. Support for Research on Safe AI Systems:

   Funding and support for research on safe, trustworthy, fair, and explainable AI algorithms and systems, as well as cybersecurity and privacy-enhancing technologies in automated systems.
8. Formal Verification and Analysis:

   Support for research on rigorous formal verification and analysis of AI systems to ensure their safety and effectiveness.

Stakeholders are encouraged to explore their own curiosities using this chatbot and the documents linked above. 

Since the original build of this chatbot, more documents like the [270-day update](https://www.whitehouse.gov/briefing-room/statements-releases/2024/07/26/fact-sheet-biden-harris-administration-announces-new-ai-actions-and-receives-additional-major-voluntary-commitment-on-ai/) on the 2023 [executive order](https://www.whitehouse.gov/briefing-room/presidential-actions/2023/10/30/executive-order-on-the-safe-secure-and-trustworthy-development-and-use-of-artificial-intelligence/) on Safe, Secure, and Trustworthy AI, have been released. 

These documents can easily be indexed for querying by our chatbot. An example of this is included in the [Dealing with Data notebook](https://github.com/dhrits/ai-ethics-bot/blob/main/nbs/Dealing%20with%20Data.ipynb). Subsequently, any new documents can be similarly incrementally indexed for querying. 



