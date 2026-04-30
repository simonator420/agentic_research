"""
vector_store.py — ChromaDB-backed cross-run memory.

Stores the best ActionPlan from each completed run, indexed by the dataset's
fingerprint vector (produced by dataset_fingerprint() in loader.py).

When a new dataset is processed, the store retrieves the top-k most similar
past configurations via cosine similarity over the fingerprint space. The
Planner Agent receives these as warm-start candidates, which reduces the
number of iterations needed to converge on a strong configuration.

The quality of retrieval improves with the number of stored runs — on the
first run for a given domain, the store is empty and the Planner starts cold.

Fingerprint vector layout (8 dimensions, see loader.py for details):
  [n_rows, n_cols, n_numeric, n_categorical,
   overall_missing_rate, target_type_enc, imbalance_ratio, duplicate_rate]

Public API
----------
VectorStore(persist_dir)
    .store_success(fingerprint, plan, dataset_name) -> None
    .retrieve_similar(fingerprint, top_k)           -> List[ActionPlan]
    .count()                                        -> int
"""

import json
import uuid
from typing import List

from src.models.schemas import ActionPlan

_COLLECTION_NAME = "pipeline_memory"


class VectorStore:
    """
    ChromaDB-backed store for cross-run memory.

    Uses cosine similarity over dataset fingerprint vectors to find structurally
    similar past datasets and return their best-performing pipeline configurations.

    Parameters
    ----------
    persist_dir : directory where ChromaDB stores its data files.
                  Must be consistent across runs for memory to accumulate.
    """

    def __init__(self, persist_dir: str = "./chroma_db"):
        try:
            import chromadb
        except ImportError:
            raise ImportError("chromadb not installed — run: pip install chromadb")

        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            # cosine distance is appropriate for fingerprint vectors of mixed scale
            metadata={"hnsw:space": "cosine"},
        )

    def store_success(
        self,
        fingerprint: List[float],
        plan: ActionPlan,
        dataset_name: str = "unknown",
    ) -> None:
        """
        Persist a successful ActionPlan keyed by the dataset's fingerprint.

        Called by the Orchestrator at the end of each run with the best plan
        found across all iterations. Multiple entries per dataset are allowed —
        retrieve_similar() returns all top-k matches regardless of origin.

        model_params is JSON-encoded because ChromaDB metadata values must be
        strings, ints, or floats — not nested dicts.
        """
        plan_meta = {
            "plan_id":           plan.plan_id,
            "imputation":        plan.imputation,
            "outlier_handling":  plan.outlier_handling,
            "encoding":          plan.encoding,
            "scaling":           plan.scaling,
            "model":             plan.model,
            "imbalance_strategy": plan.imbalance_strategy,
            "model_params":      json.dumps(plan.model_params),  # nested dict → JSON string
        }
        self._collection.add(
            ids=[str(uuid.uuid4())],
            embeddings=[fingerprint],
            documents=[dataset_name],
            metadatas=[plan_meta],
        )

    def retrieve_similar(
        self,
        fingerprint: List[float],
        top_k: int = 3,
    ) -> List[ActionPlan]:
        """
        Return up to top_k ActionPlans from the most similar past datasets.

        Returns an empty list when the store is empty so the Orchestrator
        does not need to handle the cold-start case separately — the Planner
        simply receives no memory context and generates plans from scratch.

        Results are ordered by cosine similarity (most similar first).
        """
        n_stored = self._collection.count()
        if n_stored == 0:
            return []

        results = self._collection.query(
            query_embeddings=[fingerprint],
            n_results=min(top_k, n_stored),
        )

        plans = []
        for meta in results.get("metadatas", [[]])[0]:
            plans.append(ActionPlan(
                plan_id=meta["plan_id"],
                imputation=meta["imputation"],
                outlier_handling=meta["outlier_handling"],
                encoding=meta["encoding"],
                scaling=meta["scaling"],
                model=meta["model"],
                imbalance_strategy=meta["imbalance_strategy"],
                model_params=json.loads(meta["model_params"]),
            ))

        return plans

    def count(self) -> int:
        """Return the total number of entries stored across all runs."""
        return self._collection.count()
