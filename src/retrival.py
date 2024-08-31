from haystack.utils import Secret
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.builders import PromptBuilder
from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack import Pipeline
from src.ingestion import ingestion
from src.utils import pinecone_config
import os
from dotenv import load_dotenv

load_dotenv()

prompt_template = """Answer the following query based on the provided context. If the context does
                     not include an answer, reply with 'I don't know'.\n
                     Query: {{query}}
                     Documents:
                     {% for doc in documents %}
                        {{ doc.content }}
                     {% endfor %}
                     Answer: 
                  """

def get_result(query):
    pipeline = Pipeline()

    pipeline.add_component("text_embedder",SentenceTransformersTextEmbedder(model="sentence-transformers/distiluse-base-multilingual-cased-v1"))
    pipeline.add_component("retriever",PineconeEmbeddingRetriever(document_store=pinecone_config()))
    pipeline.add_component("prompt_builder",PromptBuilder(template=prompt_template))
    pipeline.add_component("llm",HuggingFaceLocalGenerator(model="google/flan-t5-small",
                                        generation_kwargs={"max_new_tokens":250}))
    
    pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    pipeline.connect("retriever.documents","prompt_builder.documents")
    pipeline.connect("prompt_builder", "llm")

    query = query

    result=pipeline.run({"text_embedder":{"text":query},
                         "prompt_builder":{"query":query}})

    return result['llm']['replies'][0]

if __name__ == "__main__":
   
    result=get_result(query="What is RAG-Token Model ?")
    print(result)