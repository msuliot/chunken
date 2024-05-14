# chunken - CHUNK Extraction Node
import json
import sys  
import os
import re
# from glob import globp
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import helpers.openai_helper as oai 
from helpers.mongo_helper import MongoDatabase 
from helpers.pinecone_helper import Pinecone 
from msuliot.base_64 import Base64 # https://github.com/msuliot/package.utils.git

from env_config import envs
env = envs()

stop_words = set(stopwords.words('english'))

def generate_chunk_id(source, chunk_number):
    combined_input = f"{source}_chunk_{chunk_number}"
    encoded_data = Base64.encode(combined_input)
    return encoded_data


def preprocess_text(text):
    # Custom cleaning rules for headers, footers
    cleaned_text = re.sub(r'Page \d+', '', text)

    # Tag dates in various formats
    cleaned_text = re.sub(r'\d{4}-\d{2}-\d{2}', '<DATE>', cleaned_text)  # yyyy-mm-dd
    cleaned_text = re.sub(r'\b\d{2}-\d{2}-\d{2,4}\b', '<DATE>', cleaned_text)  # mm-dd-yy and mm-dd-yyyy
    month_names = "(January|February|March|April|May|June|July|August|September|October|November|December)"
    cleaned_text = re.sub(r'\b' + month_names + r' \d{1,2}, \d{4}\b', '<DATE>', cleaned_text)  # Month day, year

    # Tag phone numbers (US format; adjust regex as needed)
    cleaned_text = re.sub(r'\d{3}-\d{3}-\d{4}', '<PHONE>', cleaned_text)

    # Removing URLs or navigational links
    cleaned_text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', cleaned_text)
    
    # Tokenize text for stop word removal
    words = word_tokenize(cleaned_text)
    
    # Remove stop words and non-alphabetic tokens, preserving tagged elements
    filtered_words = [word for word in words if word.isalpha() or word in {'<DATE>', '<PHONE>'} and word not in stop_words]
    
    # Reconstruct text
    return ' '.join(filtered_words)


def read_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as config_file:
        return json.load(config_file)


def find_text_files(input_directories):
    file_paths = []
    for directory in input_directories:
        # file_paths.extend(glob(os.path.join(directory, '*.txt')))
        file_paths.extend([os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.txt')])

    return file_paths


def chunk_and_save_files(config):
    processed_files_count = 0
    total_chunks_created = 0
    oaie = oai.openai_embeddings(env.openai_key, "text-embedding-3-small")

    file_paths = find_text_files(config['input_directories'])
    chunk_size = config['chunk_size']
    extension_limit = config['chuck_extension_limit']
    namespace = config['namespace']
    database = config['database']
    print("Database:", database)
    print("Namespace:", namespace)

    for filepath in file_paths:
        print(".", end="", flush=True)
        with open(filepath, 'r', encoding='utf-8') as file:
            text = file.read()

        if not text:
            print(f"Skipping empty file: {filepath}")
            os.remove(filepath)
            continue

        if not filepath.endswith(".txt"):
            print(f"Skipping non-text file: {filepath}")
            continue

        try:
            original_filename = Base64.decode(os.path.splitext(os.path.basename(filepath))[0])
        except Exception as e:
            print(f"Skipping - Error decoding filepath: {e}")
            continue

        chunks = []
        while text:
            if len(text) <= chunk_size:
                chunks.append(text)
                break
            else:
                boundary = chunk_size
                extended_boundary = text.find('.', boundary, boundary + extension_limit)
                if extended_boundary != -1:
                    boundary = extended_boundary + 1
                else:
                    fallback_space = text.find(' ', boundary + extension_limit)
                    if fallback_space != -1:
                        boundary = fallback_space + 1
                    else:
                        boundary = boundary + extension_limit

                boundary = min(boundary, len(text))
                chunks.append(text[:boundary])
                text = text[boundary:]

        pinecone_objects = []
        mongo_objects = {
                "_id": Base64.encode(original_filename),
                "source": original_filename,
                "data":[]
            }

        for i, chunk in enumerate(chunks, 1):
            unique_chunk_id = generate_chunk_id(original_filename, i)
            values = oaie.execute(preprocess_text(chunk))

            if not values.data:
                print(f"Chunk {i} failed to create embeddings.")
                continue

            pinecone_object = {
                "id": unique_chunk_id,
                "values": values.data[0].embedding,
                "metadata": {
                    "parent_id": Base64.encode(original_filename),
                    "source": original_filename,
                    "chunk_number": i,
                }
            }

            mongo_objects["data"].append(
                {
                    "chunk_id": unique_chunk_id,
                    "chunk_number": i,
                    "text": chunk,
                }
            )

            pinecone_objects.append(pinecone_object)


        ####### TODO: upsert the objects to MongoDB
        try:
            with MongoDatabase(env.mongo_uri) as client:
                client.insert_one(database, namespace, mongo_objects)

        except Exception as e:
            print(f"Error upserting to MongoDB: {e}")
            continue
        else:
            print(f"Upserted {len(chunks)} chunks to MongoDB for {original_filename}")

            # upsert the objects to Pinecone
            pc = Pinecone(api_key=env.pinecone_key)
            index = pc.Index(database)
            index.upsert(vectors=pinecone_objects, namespace=namespace)

            processed_files_count += 1
            total_chunks_created += len(chunks)

        os.remove(filepath)
        
    return processed_files_count, total_chunks_created


if __name__ == '__main__':
    print("\nProcess started:", end=" ")
    config = read_config('config.json')
    processed_files_count, total_chunks_created = chunk_and_save_files(config)
    
    print(f"\nChunking complete. Processed {processed_files_count} files.")
    print(f"Total chunked files created: {total_chunks_created}")
