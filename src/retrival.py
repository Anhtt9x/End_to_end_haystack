from haystack.utils import Secret
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.builders import PromptBuilder
from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever
from haystack.components.generators import HuggingFaceAPIGenerator
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

def get_result(query,documents_store):
    pipeline = Pipeline()

    pipeline.add_component("text_embedder",SentenceTransformersTextEmbedder())
    pipeline.add_component("retriever",PineconeEmbeddingRetriever(documents_store))
    pipeline.add_component("prompt_builder",PromptBuilder(template=prompt_template))
    pipeline.add_component("llm",HuggingFaceAPIGenerator())


if __name__ == "__main__":
    get_result()