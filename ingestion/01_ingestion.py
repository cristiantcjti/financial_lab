import os
import uuid

from dotenv import load_dotenv
from fastembed import (
    LateInteractionTextEmbedding,
    SparseTextEmbedding,
    TextEmbedding,
)
from qdrant_client import QdrantClient, models
from utils.semantic_chunker import SemanticChunker

load_dotenv()


DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SPARSE_MODEL = "Qdrant/bm25"
COLBERT_MODEL = "colbert-ir/colbertv2.0"
COLLECTION_NAME = "financial"
FILE_PATH = "./AAPL_10-K_1A_temp.md"
MAX_TOKENS = 300

qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY")
)


qdrant.delete_collection(COLLECTION_NAME)
qdrant.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        "dense": models.VectorParams(
            size=384, distance=models.Distance.COSINE
        ),
        "colbert": models.VectorParams(
            size=128,
            distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM
            ),
        ),
    },
    sparse_vectors_config={"sparse": models.SparseVectorParams()},
)


with open(FILE_PATH, encoding="utf-8") as f:
    content = f.read()


chunker = SemanticChunker(max_tokens=MAX_TOKENS)
chunks = chunker.create_chunks(content)


dense_model = TextEmbedding(DENSE_MODEL)
sparse_model = SparseTextEmbedding(SPARSE_MODEL)
colbert_model = LateInteractionTextEmbedding(COLBERT_MODEL)

points = []
for chunk in chunks:
    dense_embedding = list(dense_model.passage_embed([chunk]))[0].tolist()
    sparse_obj = list(sparse_model.passage_embed([chunk]))[0].as_object()
    sparse_embedding = models.SparseVector(
        indices=sparse_obj["indices"].tolist(),
        values=sparse_obj["values"].tolist(),
    )
    colbert_embedding = list(colbert_model.passage_embed([chunk]))[0].tolist()

    point = models.PointStruct(
        id=str(uuid.uuid4()),
        vector={
            "dense": dense_embedding,
            "sparse": sparse_embedding,
            "colbert": colbert_embedding,
        },
        payload={"text": chunk, "source": FILE_PATH},
    )
    points.append(point)

qdrant.upload_points(collection_name=COLLECTION_NAME, points=points)

query_text = "What are the main financial risks?"
query_dense = list(dense_model.query_embed([query_text]))[0].tolist()
query_sparse_obj = list(sparse_model.query_embed([query_text]))[0].as_object()
query_sparse = models.SparseVector(
    indices=query_sparse_obj["indices"].tolist(),
    values=query_sparse_obj["values"].tolist(),
)
query_colbert = list(colbert_model.query_embed([query_text]))[0].tolist()


results = qdrant.query_points(
    collection_name=COLLECTION_NAME,
    prefetch=[
        models.Prefetch(
            prefetch=[
                models.Prefetch(query=query_dense, using="dense", limit=10),
                models.Prefetch(query=query_sparse, using="sparse", limit=10),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=20,
        )
    ],
    query=query_colbert,
    using="colbert",
    limit=3,
)

max_score = max(point.score for point in results.points)

for r in results.points:
    normalized_score = r.score / max_score
    print(f"Score: {normalized_score}")
    if r.payload:
        print(f"Text: {r.payload['text'][:100]}...")
    print("-" * 80)
