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

    # Function to upsert data
    def upsert_data_df(self, df, create_if_missing):
        print("Start: Upserting data to Pinecone index")
        have_index = self.set_pinecone_index()

        if not have_index and not create_if_missing:
            log.error(f"Index {self.index_name} not found, and create_if_missing is False.")
            return False
        elif not have_index and create_if_missing:
            created = self.create_pinecone_index()
            if not created:
                log.error("error creating index")
                return False
            else:
                self.set_pinecone_index()
        elif have_index:
            log.info(f"Index {self.index_name} already exists.")
            pass
        else:
            log.error("Error setting Pinecone index.")
            return False
        
        
        log.info(f"Start: Upserting data to Pinecone index {self.index_name}")
        prepped = []

        for i, row in tqdm(df.iterrows(), total=df.shape[0], desc="Upserting data"):
            # meta = ast.literal_eval(row['metadata'])
            meta = row['metadata']
            prepped.append({'id': str(row['id']), 
                            'values': row['values'],
                            'metadata': meta})
            if len(prepped) >= 200: # batching upserts
                self.index.upsert(prepped)
                prepped = []

        # Upsert any remaining entries after the loop
        if len(prepped) > 0:
            self.index.upsert(prepped)
        
        
        log.info(f"Done: Upserting data to Pinecone index {self.index_name} with {len(df)} rows")
        print(f"Done: Upserting data to Pinecone index {self.index_name} with {len(df)} rows")
        return True


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
