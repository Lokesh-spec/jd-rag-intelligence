from enum import Enum


class RetrieverType(str, Enum):
    """Supported JD retrieval strategies."""

    SEMANTIC = "semantic"
    THRESHOLD = "threshold"
    MMR = "mmr"
    BM25 = "bm25"
    HYBRID = "hybrid"
    COMPRESSED = "compressed"

    @classmethod
    def from_string(cls, value: str) -> "RetrieverType":
        """Parse user-facing labels (CLI, notebook, legacy names) into enum values."""
        normalized = value.lower().strip().replace("-", "_").replace(" ", "_")
        aliases: dict[str, RetrieverType] = {
            "semantic": cls.SEMANTIC,
            "semantic_search": cls.SEMANTIC,
            "similarity": cls.SEMANTIC,
            "threshold": cls.THRESHOLD,
            "similarity_score_threshold": cls.THRESHOLD,
            "semantic_search_with_threshold": cls.THRESHOLD,
            "semantic_search_with_thershold": cls.THRESHOLD,
            "mmr": cls.MMR,
            "bm25": cls.BM25,
            "hybrid": cls.HYBRID,
            "ensemble": cls.HYBRID,
            "compressed": cls.COMPRESSED,
            "compression": cls.COMPRESSED,
        }
        try:
            return aliases[normalized]
        except KeyError as exc:
            supported = ", ".join(sorted({m.value for m in cls}))
            raise ValueError(
                f"Unknown retriever type '{value}'. Supported: {supported}"
            ) from exc
