# Key Documents

1. [Report](./Report.md) outlining various parts of this project as well as **answers to various questions posed**.
2. [Dealing with Data](nbs/Dealing%20with%20Data.ipynb) notebook which handles initial and revisited chunking and indexing of data in a qdrant cloud database. It also describes the chunking strategies. 
3. [Test Data and RAGAS Evaluation](./nbs/Test%20Data%20and%20RAGAS%20Evaluation.ipynb) notebook which handles the generation of a Golden Evaluation Testset as well as evaluation of two chunking strategies using RAGAS. 
4. [Finetuning](./nbs/Fine_Tuning_nomic_embed_text_v1_on_AI_Ethics_Docs.ipynb) notebook which finetunes a version of `nomic-ai/nomic-embed-text-v1` embedding model for this task. 
5. The [Finetuning](./nbs/Fine_Tuning_nomic_embed_text_v1_on_AI_Ethics_Docs.ipynb) notebook also contains an evaluation of a chunkings strategy based on this finetuned model and comparisons to previous strategies using RAGAS framework. 
6. [Managing Expectations](./Report.md) section included in the report which summarizes key conclusions. 
7. [Github](https://github.com/dhrits/ai-ethics-bot) repository with all the code. 
8. [Chatbot](https://huggingface.co/spaces/deman539/ai-ethics-bot) deployed to huggingface spaces.
9. [Loom video](https://www.loom.com/share/675c36ee837e4f44a5a9f6cc03891cc8?sid=a07096c0-0be4-4c9c-ac43-611389f5b712) describing key functionality of the chatbot