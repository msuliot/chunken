from pinecone import Pinecone, ServerlessSpec
from tqdm.auto import tqdm
import ast
import os
import sys
import json

### Logging ###
import logging as log


class pinecone_logic:

    def __init__(self, pinecone_api_key, index_name):
        log.info(f"CLASS:pinecone_logic initialized client with index name: {index_name}")
        try:
            self.pinecone = Pinecone(api_key=pinecone_api_key)
            self.index_name = index_name
            self.index = None
        except Exception as e:
            log.error(f"Error initializing Pinecone client: {e}")
            print(f"Error initializing Pinecone client: {e}")
            sys.exit(1)

    def delete_pinecone_index(self):
        try:
            self.pinecone.delete_index(self.index_name)
            self.index = None
            log.info(f"Index '{self.index_name}' successfully deleted.")
        except Exception as e:
            log.error(f"Error deleting index '{self.index_name}': {e}")


    def create_pinecone_index(self):
        self.pinecone.create_index(
            name=self.index_name, 
            dimension=1536, 
            metric='cosine', 
            spec=ServerlessSpec(cloud='aws', region='us-west-2'))
            
        self.index = self.pinecone.Index(self.index_name)
        log.info(f"Index {self.index_name} created successfully.")
        return True


    def set_pinecone_index(self):
        try:
            if self.index_name in [index.name for index in self.pinecone.list_indexes()]:
                self.index = self.pinecone.Index(self.index_name)
                log.info(f"Setting Pinecone index to {self.index_name}")
                return True
            else:
                log.warning(f"Index {self.index_name} does not exist.")
                return False
        except Exception as e:
            log.error(f"Error setting Pinecone index: {e}")
            print(f"Error setting Pinecone index. Please check log file for details.")
            sys.exit(1)


    def search_pinecone_index(self, embed, top):
        have_index = self.set_pinecone_index()
        if not have_index:
            log.error(f"Index {self.index_name} not found, please create index first.")
            return None
        try:    
            result = self.index.query(vector=embed.data[0].embedding, top_k=top, include_metadata=True)
        except Exception as e:
            log.error(f"Error querying Pinecone index: {e}")
            return None

        return result
    
    def display_text_from_index_search(self, data):
        for match in data['matches']:
            print(match['metadata'])
            print("\n")
            # print('-' * 80)  # Print a line separator for readability
