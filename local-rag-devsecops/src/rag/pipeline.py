"""
Pipeline RAG complet : orchestration retriever + LLM.

Phase 2 du RAG :
1. Recherche sémantique des chunks pertinents (retriever)
2. Construction d'un prompt augmenté avec citations
3. Génération de la réponse via Ollama (LLM)
4. Retour de la réponse + sources utilisées
"""
from dataclasses import dataclass, field
from typing import Iterator, Optional

from rag.llm import OllamaClient, LLMError
from rag.retriever import Retriever, RetrievedChunk, RetrieverError


# Configuration par défaut
DEFAULT_TOP_K = 5
DEFAULT_SIMILARITY_THRESHOLD = 0.3
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 800


# Le system prompt : définit le rôle et les contraintes du LLM
SYSTEM_PROMPT = """Tu es un assistant DevSecOps spécialisé en sécurité IA.
Tu réponds UNIQUEMENT à partir des sources fournies dans le contexte.

RÈGLES STRICTES :
1. Si la réponse n'est pas dans les sources, réponds exactement :
   "Je n'ai pas cette information dans la base documentaire."
2. Cite tes sources entre crochets, ex: [Source: falco_runbook.md]
3. Réponds en français, sauf si la question est en anglais.
4. Sois concis, factuel, technique.
5. Ne fais JAMAIS d'inférence au-delà de ce qui est explicitement écrit dans les sources.
6. Ne mélange pas les informations de différentes sources sauf si elles sont complémentaires."""


@dataclass
class RAGResponse:
    """Réponse complète d'une requête RAG."""
    answer: str
    sources: list[RetrievedChunk] = field(default_factory=list)
    num_chunks_used: int = 0
    no_context_found: bool = False


class RAGPipeline:
    """
    Pipeline RAG : recherche + génération.
    
    Instancie une seule fois (charge le modèle d'embedding et le client Ollama),
    puis utilise .ask() ou .ask_stream() autant que tu veux.
    """
    
    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        llm_client: Optional[OllamaClient] = None,
    ):
        """
        Args:
            retriever: Instance pré-configurée (sinon créée avec defaults).
            llm_client: Instance pré-configurée (sinon créée avec defaults).
        """
        self.retriever = retriever or Retriever()
        self.llm = llm_client or OllamaClient()
    
    # ---------- Méthodes publiques ----------
    
    def ask(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> RAGResponse:
        """
        Mode bloquant : retourne la réponse complète.
        
        Utile pour : tests automatisés, batch processing, scripts.
        """
        chunks = self._retrieve(question, top_k, similarity_threshold)
        
        if not chunks:
            return RAGResponse(
                answer="Je n'ai pas cette information dans la base documentaire.",
                sources=[],
                num_chunks_used=0,
                no_context_found=True,
            )
        
        user_prompt = self._build_user_prompt(question, chunks)
        
        try:
            response = self.llm.generate(
                prompt=user_prompt,
                system=SYSTEM_PROMPT,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except LLMError as e:
            raise RAGError(f"Erreur LLM : {e}")
        
        return RAGResponse(
            answer=response.content,
            sources=chunks,
            num_chunks_used=len(chunks),
            no_context_found=False,
        )
    
    def ask_stream(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> tuple[Iterator[str], list[RetrievedChunk]]:
        """
        Mode streaming : retourne un itérateur de tokens + la liste des sources.
        
        Utile pour : CLI interactive, UI temps réel.
        
        Returns:
            Tuple (itérateur de tokens, liste des chunks utilisés).
            Si aucun chunk trouvé, l'itérateur yield un message d'excuse.
        """
        chunks = self._retrieve(question, top_k, similarity_threshold)
        
        if not chunks:
            def empty_stream():
                yield "Je n'ai pas cette information dans la base documentaire."
            return empty_stream(), []
        
        user_prompt = self._build_user_prompt(question, chunks)
        
        try:
            stream = self.llm.stream(
                prompt=user_prompt,
                system=SYSTEM_PROMPT,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except LLMError as e:
            raise RAGError(f"Erreur LLM : {e}")
        
        return stream, chunks
    
    # ---------- Méthodes privées ----------
    
    def _retrieve(
        self,
        question: str,
        top_k: int,
        similarity_threshold: float,
    ) -> list[RetrievedChunk]:
        """Wrapper autour du retriever avec gestion d'erreurs."""
        try:
            return self.retriever.search(
                query=question,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
            )
        except RetrieverError as e:
            raise RAGError(f"Erreur retriever : {e}")
    
    def _build_user_prompt(
        self,
        question: str,
        chunks: list[RetrievedChunk],
    ) -> str:
        """Construit le prompt augmenté avec les sources."""
        # Formater les sources
        sources_blocks = []
        for chunk in chunks:
            block = (
                f"[Source: {chunk.source}, "
                f"chunk {chunk.chunk_index + 1}/{chunk.total_chunks}]\n"
                f"{chunk.content}"
            )
            sources_blocks.append(block)
        
        context = "\n\n---\n\n".join(sources_blocks)
        
        return f"""Sources disponibles :

{context}

---

Question : {question}

Réponds à la question en citant les sources entre crochets."""


class RAGError(Exception):
    """Erreur durant le pipeline RAG."""
    pass