import warnings
from collections import defaultdict

import hdbscan
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

warnings.simplefilter(action="ignore", category=FutureWarning)


class SemanticChunker:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        min_cluster_size: int = 3,
        orphan_cluster_size: int = 2,
        max_tokens: int = 300,
    ) -> None:
        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = 512
        self.min_cluster_size = min_cluster_size
        self.orphan_cluster_size = orphan_cluster_size
        self.max_tokens = max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _cluster_and_process(
        self, texts: list[str], min_size: int
    ) -> tuple[list[str], list[str]]:
        if len(texts) <= 1:
            return texts, texts if len(texts) == 1 else []

        embeddings = self.model.encode(texts, show_progress_bar=False)

        labels = hdbscan.HDBSCAN(
            min_cluster_size=min_size, metric="euclidean"
        ).fit_predict(embeddings)

        clusters = defaultdict(list)
        orphans = []
        for i, label in enumerate(labels):
            if label != -1:
                clusters[label].append(texts[i])
            else:
                orphans.append(texts[i])

        chunks = []
        for cluster_paragraphs in clusters.values():
            current_chunk = []
            current_tokens = 0

            for paragraph in cluster_paragraphs:
                paragraph_tokens = len(
                    self.tokenizer.encode(paragraph, add_special_tokens=False)
                )

                if (
                    current_tokens + paragraph_tokens > self.max_tokens
                    and current_chunk
                ):
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = [paragraph]
                    current_tokens = paragraph_tokens
                else:
                    current_chunk.append(paragraph)
                    current_tokens += paragraph_tokens

            if current_chunk:
                chunks.append("\n\n".join(current_chunk))

        return chunks, orphans

    def create_chunks(self, text_content: str) -> list[str]:
        paragraphs = [
            p.strip()
            for p in text_content.split("\n")
            if len(p.strip().split()) > 10
        ]

        if not paragraphs:
            return []

        final_chunks, orphans = self._cluster_and_process(
            texts=paragraphs, min_size=self.min_cluster_size
        )

        if len(orphans) > 1:
            orphan_chunks, single_orphans = self._cluster_and_process(
                texts=orphans, min_size=self.orphan_cluster_size
            )
            final_chunks.extend(orphan_chunks)
            final_chunks.extend(single_orphans)
        elif orphans:
            final_chunks.extend(orphans)

        return final_chunks
