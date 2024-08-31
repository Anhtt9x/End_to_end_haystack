from haystack import Pipeline
from haystack.components.writers import DocumentWriter
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore
from haystack.components.converters import PyPDFToDocument
from pathlib import Path
import os
from dotenv import load_dotenv
from src.utils import pinecone_config

load_dotenv()

def ingestion(document_store):
    pipeline = Pipeline()

    pipeline.add_component("converter",PyPDFToDocument())
    pipeline.add_component("splitter",DocumentSplitter(split_by="sentence",split_length=20))
    pipeline.add_component("embedder",SentenceTransformersDocumentEmbedder())
    pipeline.add_component("writer",DocumentWriter(document_store=document_store))

    pipeline.connect("converter","splitter")
    pipeline.connect("splitter","embedder")
    pipeline.connect("embedder","writer")

    pipeline.run({"converter":{"sources":[Path("data/Retrieval-Augmented-Generation-for-NLP.pdf")]}})

if __name__ == "__main__":
    document_store=pinecone_config()    

    ingestion(document_store)