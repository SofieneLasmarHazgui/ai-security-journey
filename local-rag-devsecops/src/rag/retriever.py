"""
Module de recherche sémantique dans ChromaDB.

Phase 2 du RAG : étant donné une question, retrouve les chunks
les plus pertinents avec leurs métadonnées pour citation.
"""
from dataclasses import dataclass
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


# Valeurs par défaut alignées avec ingest.py
DEFAULT_TOP_K = 5
DEFAULT_SIMILARITY_THRESHOLD = 0.3
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_CHROMA_PATH = "./vectordb"
DEFAULT_COLLECTION = "devsecops_docs"


@dataclass
class RetrievedChunk:
    """Un chunk retrouvé par la recherche sémantique."""
    content: str
    source: str
    chunk_index: int
    total_chunks: int
    doc_type: str
    similarity: float    # score 0..1 (1 = identique)
    content_hash: str


class RetrieverError(Exception):
    """Erreur durant la recherche."""
    pass


class Retriever:
    """
    Wrapper autour de ChromaDB pour la recherche sémantique.
    
    Charge une seule fois le modèle d'embedding et le client ChromaDB,
    puis permet de faire des recherches multiples efficacement.
    """
    
    def __init__(
        self,
        chroma_path: str = DEFAULT_CHROMA_PATH,
        collection_name: str = DEFAULT_COLLECTION,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    ):
        """Initialise le retriever (modèle + ChromaDB)."""
        self.embedding_model_name = embedding_model_name
        self.collection_name = collection_name
        
        # Charger le modèle d'embedding
        self._model = SentenceTransformer(embedding_model_name)
        
        # Connexion ChromaDB
        self._client = chromadb.PersistentClient(
            path=chroma_path,
            settings=Settings(anonymized_telemetry=False),
        )
        
        # Récupérer la collection (doit exister, sinon erreur explicite)
        try:
            self._collection = self._client.get_collection(name=collection_name)
        except Exception as e:
            raise RetrieverError(
                f"Collection '{collection_name}' introuvable dans {chroma_path}. "
                f"As-tu lancé l'indexation avant ? Détails : {e}"
            )
    
    @property
    def collection_size(self) -> int:
        """Nombre total de chunks indexés."""
        return self._collection.count()
    
    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        filter_metadata: Optional[dict] = None,
    ) -> list[RetrievedChunk]:
        """
        Recherche les chunks les plus pertinents pour une question.
        
        Args:
            query: La question de l'utilisateur (texte libre).
            top_k: Nombre maximum de chunks à retourner.
            similarity_threshold: Seuil minimal de similarité (0..1).
                Les chunks en dessous sont écartés.
            filter_metadata: Filtre ChromaDB optionnel sur les metadonnées.
                Exemple : {"doc_type": "pdf"}
        
        Returns:
            Liste de RetrievedChunk triés par similarité décroissante.
        """
        if not query.strip():
            raise RetrieverError("La requête est vide.")
        
        if self.collection_size == 0:
            raise RetrieverError(
                "La collection est vide. Aucun document à interroger."
            )
        
        # 1) Embedder la question avec le même modèle que l'indexation
        query_embedding = self._model.encode(
            [query],
            show_progress_bar=False,
            convert_to_numpy=True,
        ).tolist()
        
        # 2) Recherche dans ChromaDB
        # On demande plus que top_k pour avoir de la marge après filtrage
        n_results = min(top_k * 2, self.collection_size)
        
        query_kwargs = {
            "query_embeddings": query_embedding,
            "n_results": n_results,
        }
        if filter_metadata:
            query_kwargs["where"] = filter_metadata
        
        results = self._collection.query(**query_kwargs)
        
        # 3) Parser les résultats et appliquer le threshold
        retrieved: list[RetrievedChunk] = []
        
        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        
        for i in range(len(ids)):
            # ChromaDB retourne une distance ; on la convertit en similarité
            # Pour la métrique cosinus, similarity = 1 - distance
            similarity = max(0.0, 1.0 - distances[i])
            
            if similarity < similarity_threshold:
                continue
            
            metadata = metadatas[i]
            retrieved.append(RetrievedChunk(
                content=documents[i],
                source=metadata.get("source", "unknown"),
                chunk_index=metadata.get("chunk_index", 0),
                total_chunks=metadata.get("total_chunks", 0),
                doc_type=metadata.get("doc_type", "unknown"),
                similarity=similarity,
                content_hash=metadata.get("content_hash", ""),
            ))
            
            if len(retrieved) >= top_k:
                break
        
        return retrieved
    
    def search_with_context(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> str:
        """
        Recherche et formate le contexte directement utilisable
        dans un prompt LLM.
        
        Format de sortie (prêt à injecter dans un prompt) :
        
            [Source: file.md, chunk 3/12]
            Texte du chunk...
            
            [Source: file.md, chunk 5/12]
            Texte du chunk...
        
        Args:
            query: Question utilisateur.
            top_k: Nombre max de chunks.
            similarity_threshold: Seuil de filtrage.
        
        Returns:
            String formatée pour injection dans un prompt LLM.
        """
        chunks = self.search(query, top_k, similarity_threshold)
        
        if not chunks:
            return "[Aucun document pertinent trouvé.]"
        
        formatted_blocks = []
        for chunk in chunks:
            block = (
                f"[Source: {chunk.source}, chunk {chunk.chunk_index + 1}/{chunk.total_chunks}, "
                f"similarity: {chunk.similarity:.2f}]\n"
                f"{chunk.content}"
            )
            formatted_blocks.append(block)
        
        return "\n\n---\n\n".join(formatted_blocks)