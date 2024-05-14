from pinecone import Pinecone
import sys  

api = "c3b02886-8a90-4ddb-9a0c-4554f479986e"
database = "blades-of-grass"
namespace = "demo24"
parent_id = "L1VzZXJzL21zdWxpb3QvRGVza3RvcC9tc3VsaW90L2xpdmluZ3RvMTAwLnBkZg=="
last_chuck_id = "L1VzZXJzL21zdWxpb3QvRGVza3RvcC9tc3VsaW90L2xpdmluZ3RvMTAwLnBkZl9jaHVua18z"
last_chunk_number = 15

pc = Pinecone(api_key=api)
index = pc.Index(database)

# Query to filter by metadata
query_response = index.query(
    id=last_chuck_id,
    namespace=namespace,
    filter={
        "parent_id": {"$eq": parent_id},
        "chunk_number": {"$gt": last_chunk_number}
    },
    top_k=100,
    include_metadata=False,
    include_values=False
)

# Extract the IDs of the matches of any orphan chunks
if len(query_response.matches) == 0:
    print("No orphaned chunks found.")
    sys.exit()

ids = []
for match in query_response.matches:
    ids.append(match.id)

query_del = index.delete(
    namespace=namespace,
    ids=ids
)