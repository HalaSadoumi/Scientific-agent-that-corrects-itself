from types import SimpleNamespace

from agent.retrieval.embeddings import create_embeddings
from agent.retrieval.graph import extract_entities
from agent.retrieval.routing import choose_route


def test_router_chooses_graph_hybrid_for_relational_astronomy_question() -> None:
    route = choose_route(
        "How is NGC 123 related to Chandra observations?", embeddings=True, graph=True
    )
    assert route.method == "graph_hybrid"
    assert "links" in route.reason


def test_router_uses_keyword_fallback_without_optional_services() -> None:
    route = choose_route("stellar mass", embeddings=False, graph=False)
    assert route.method == "sqlite_fts5_bm25"
    assert "keyword" in route.reason


def test_entity_extraction_is_astronomy_specific_and_deduplicated() -> None:
    assert extract_entities("NGC 123 was observed by Chandra; NGC 123 is bright.") == [
        "NGC 123",
        "Chandra",
    ]


def test_embedding_backend_none_does_not_call_a_provider() -> None:
    settings = SimpleNamespace(embedding_backend="none")
    assert create_embeddings(["a passage"], settings) is None
