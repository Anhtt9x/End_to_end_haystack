from haystack_integrations.document_stores.pinecone import PineconeDocumentStore
import os
from dotenv import load_dotenv



load_dotenv()



def pinecone_config():

    documents_store=PineconeDocumentStore(index="end-to-end-haystack",dimension=512)

    return documents_store