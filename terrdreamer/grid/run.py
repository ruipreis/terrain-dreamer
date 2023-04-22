from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


def recreate_collection(client, collection_name, dim: int, distance=Distance.COSINE):
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dim, distance=distance),
    )


def get_client():
    return QdrantClient(host="localhost", port=6333)


# TO setup run

# docker pull qdrant/qdrant
# docker run -p 6333:6333 qdrant/qdrant
