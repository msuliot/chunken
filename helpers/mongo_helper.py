from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import certifi


class MongoDatabase:
    _mongo_client = None  # Class-level attribute to hold the shared MongoClient instance

    @classmethod
    def get_mongo_client(cls, uri):
        if cls._mongo_client is None:
            cls._mongo_client = MongoClient(uri, server_api=ServerApi('1'), tlsCAFile=certifi.where())
        return cls._mongo_client

    def str_to_bool(self, s):
        return s.lower() in ['true', '1', 't', 'y', 'yes']

    def __init__(self, mongo_uri):
        # print(f"Connecting to MongoDB at {mongo_uri}")
        self.mongo_client = MongoDatabase.get_mongo_client(mongo_uri)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # No need to close the client here as it's shared across the application
        if exc_type:
            print(f"An exception occurred: {exc_value}")

    def find_one(self, database_name, collection_name, api_key_query, projection={}):
        db = self.mongo_client[database_name]
        collection = db[collection_name]
        document = collection.find_one(api_key_query, projection)
        return document

    def find(self, database_name, collection_name, query, projection={}, sort=None):
        db = self.mongo_client[database_name]
        collection = db[collection_name]
        documents = collection.find(query, projection)
        if sort is not None:
            documents = documents.sort(sort)
        return list(documents)

    def insert_one(self, database_name, collection_name, document):
        db = self.mongo_client[database_name]
        collection = db[collection_name]
        insert_result = collection.insert_one(document)
        return insert_result.inserted_id

    def update_one(self, database_name, collection_name, query, update, upsert=False):
        db = self.mongo_client[database_name]
        collection = db[collection_name]
        update_result = collection.update_one(query, update, upsert=upsert)
        return update_result

    def insert_or_update_chunk(self, database_name, collection_name, mongo_objects):
        query = {"_id": mongo_objects["_id"]}

        update = {
            "$set": {
                "source": mongo_objects["source"],
                "data": mongo_objects["data"]
            },
        }

        update_result = self.update_one(database_name, collection_name, query, update, upsert=True)

        if update_result.matched_count > 0:
            print("UPDATED Mongo with ID:", mongo_objects["_id"])
        elif update_result.upserted_id is not None:
            print("INSERTED Mongo with ID:", update_result.upserted_id)
        else:
            print(f"This should not happen. source: {mongo_objects['source']}")

        return update_result
    

    def get_document_by_chunk_id(self, database_name, collection_name, chunk_id):
        pipeline = [
            {
                "$match": {
                    "data.chunk_id": chunk_id
                }
            },
            {
                "$project": {
                    "data": {
                        "$filter": {
                            "input": "$data",
                            "as": "chunk",
                            "cond": {"$eq": ["$$chunk.chunk_id", chunk_id]}
                        }
                    }
                }
            }
        ]

        db = self.mongo_client[database_name]
        collection = db[collection_name]
        result = collection.aggregate(pipeline)
        return list(result)
