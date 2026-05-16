"""
Module d'indexation des documents pour le RAG.

Pipeline :
1. Lecture des documents (PDF, Markdown, TXT)
2. Chunking récursif avec overlap
3. Génération d'embeddings
4. Stockage dans ChromaDB avec métadonnées de traçabilité
"""
import hashlib
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# Configuration par défaut (peut être surchargée via .env)
DEFAULT_CHUNK_SIZE = 800        # caractères par chunk
DEFAULT_CHUNK_OVERLAP = 100     # caractères de chevauchement
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_CHROMA_PATH = "./vectordb"
DEFAULT_COLLECTION = "devsecops_docs"

# Types de fichiers supportés
SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt"}


@dataclass
class Chunk:
    """Représente un chunk indexable."""
    content: str
    source: str            # nom du fichier d'origine
    chunk_index: int       # position du chunk dans le document
    total_chunks: int      # nombre total de chunks dans le document
    doc_type: str          # pdf, markdown, txt
    content_hash: str      # SHA-256 du contenu (intégrité)


class IngestError(Exception):
    """Erreur durant l'indexation."""
    pass


# ---------- Lecture des documents ----------

def read_document(file_path: Path) -> str:
    """
    Lit un document et retourne son contenu texte.
    
    Args:
        file_path: Chemin vers le fichier.
    
    Returns:
        Contenu textuel du document.
    
    Raises:
        IngestError: Si le format n'est pas supporté ou si la lecture échoue.
    """
    ext = file_path.suffix.lower()
    
    if ext not in SUPPORTED_EXTENSIONS:
        raise IngestError(
            f"Format non supporté : {ext}. Supportés : {SUPPORTED_EXTENSIONS}"
        )
    
    try:
        if ext == ".pdf":
            return _read_pdf(file_path)
        elif ext in {".md", ".txt"}:
            return file_path.read_text(encoding="utf-8")
    except Exception as e:
        raise IngestError(f"Erreur de lecture {file_path.name} : {e}")


def _read_pdf(file_path: Path) -> str:
    """Extrait le texte d'un PDF page par page."""
    reader = PdfReader(file_path)
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n\n".join(pages)


# ---------- Chunking ----------

def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """
    Découpe un texte en chunks avec chevauchement.
    
    Algorithme : recursive character splitting.
    Tente de couper à des frontières naturelles dans cet ordre :
    1. Double saut de ligne (paragraphe)
    2. Saut de ligne simple
    3. Point + espace
    4. Virgule + espace
    5. Caractère brut (fallback)
    
    Args:
        text: Texte à découper.
        chunk_size: Taille cible d'un chunk (caractères).
        overlap: Chevauchement entre chunks adjacents.
    
    Returns:
        Liste de chunks.
    """
    if not text.strip():
        return []
    
    # Séparateurs par ordre de priorité
    separators = ["\n\n", "\n", ". ", ", ", " "]
    
    chunks: list[str] = []
    
    def _split_recursive(text: str, sep_idx: int = 0) -> list[str]:
        """Split récursif en essayant chaque séparateur."""
        if len(text) <= chunk_size:
            return [text]
        
        if sep_idx >= len(separators):
            # Fallback : coupe brute
            return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        
        sep = separators[sep_idx]
        parts = text.split(sep)
        
        # Regrouper les parts pour atteindre la taille cible
        result = []
        current = ""
        
        for part in parts:
            candidate = current + sep + part if current else part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    result.append(current)
                # Si la part seule est trop grosse, on récursive avec le séparateur suivant
                if len(part) > chunk_size:
                    result.extend(_split_recursive(part, sep_idx + 1))
                    current = ""
                else:
                    current = part
        
        if current:
            result.append(current)
        
        return result
    
    raw_chunks = _split_recursive(text)
    
    # Ajouter l'overlap
    if overlap > 0 and len(raw_chunks) > 1:
        chunks_with_overlap = [raw_chunks[0]]
        for i in range(1, len(raw_chunks)):
            previous_tail = raw_chunks[i - 1][-overlap:]
            chunks_with_overlap.append(previous_tail + raw_chunks[i])
        return chunks_with_overlap
    
    return raw_chunks


def build_chunks(file_path: Path) -> list[Chunk]:
    """
    Lit un fichier et le découpe en chunks avec métadonnées.
    
    Args:
        file_path: Chemin du document.
    
    Returns:
        Liste d'objets Chunk prêts à indexer.
    """
    text = read_document(file_path)
    raw_chunks = chunk_text(text)
    
    ext = file_path.suffix.lower()
    doc_type = {
        ".pdf": "pdf",
        ".md": "markdown",
        ".txt": "text",
    }.get(ext, "unknown")
    
    chunks = []
    for i, content in enumerate(raw_chunks):
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        chunks.append(Chunk(
            content=content,
            source=file_path.name,
            chunk_index=i,
            total_chunks=len(raw_chunks),
            doc_type=doc_type,
            content_hash=content_hash,
        ))
    
    return chunks


# ---------- Indexation dans ChromaDB ----------

def get_chroma_client(persist_path: str = DEFAULT_CHROMA_PATH) -> chromadb.PersistentClient:
    """Initialise un client ChromaDB persistant avec télémétrie désactivée."""
    return chromadb.PersistentClient(
        path=persist_path,
        settings=Settings(anonymized_telemetry=False),
    )


def get_or_create_collection(
    client: chromadb.PersistentClient,
    name: str = DEFAULT_COLLECTION,
) -> chromadb.Collection:
    """Récupère la collection ou la crée si elle n'existe pas."""
    return client.get_or_create_collection(
        name=name,
        metadata={"description": "DevSecOps knowledge base for local RAG"},
    )


def index_directory(
    docs_dir: Path,
    collection: chromadb.Collection,
    embedding_model: SentenceTransformer,
) -> dict:
    """
    Indexe tous les documents supportés d'un dossier.
    
    Args:
        docs_dir: Chemin du dossier à indexer.
        collection: Collection ChromaDB cible.
        embedding_model: Modèle d'embedding chargé.
    
    Returns:
        Statistiques d'indexation : {nb_docs, nb_chunks, skipped}.
    """
    if not docs_dir.exists():
        raise IngestError(f"Le dossier {docs_dir} n'existe pas.")
    
    # Lister les fichiers supportés (récursif)
    files = [
        f for f in docs_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        and f.name != "README.md"  # on ignore le README du dossier
    ]
    
    if not files:
        return {"nb_docs": 0, "nb_chunks": 0, "skipped": 0}
    
    print(f"📚 {len(files)} document(s) à indexer\n")
    
    total_chunks = 0
    skipped = 0
    ingested_at = datetime.now(timezone.utc).isoformat()
    
    for file_path in tqdm(files, desc="Indexation"):
        try:
            chunks = build_chunks(file_path)
            if not chunks:
                skipped += 1
                continue
            
            # Préparation pour ChromaDB
            ids = [f"{file_path.name}::{c.chunk_index}" for c in chunks]
            documents = [c.content for c in chunks]
            metadatas = [
                {
                    "source": c.source,
                    "chunk_index": c.chunk_index,
                    "total_chunks": c.total_chunks,
                    "doc_type": c.doc_type,
                    "content_hash": c.content_hash,
                    "ingested_at": ingested_at,
                }
                for c in chunks
            ]
            
            # Génération des embeddings (en batch pour la performance)
            embeddings = embedding_model.encode(
                documents,
                show_progress_bar=False,
                convert_to_numpy=True,
            ).tolist()
            
            # Upsert : remplace si déjà indexé (idempotent)
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            
            total_chunks += len(chunks)
        
        except IngestError as e:
            print(f"⚠️  Skipping {file_path.name} : {e}")
            skipped += 1
    
    return {
        "nb_docs": len(files) - skipped,
        "nb_chunks": total_chunks,
        "skipped": skipped,
    }


# ---------- Fonction haut niveau ----------

def ingest(
    docs_dir: str = "./docs",
    chroma_path: str = DEFAULT_CHROMA_PATH,
    collection_name: str = DEFAULT_COLLECTION,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> dict:
    """
    Point d'entrée haut niveau pour indexer un dossier de documents.
    
    Args:
        docs_dir: Dossier contenant les documents.
        chroma_path: Chemin de persistance ChromaDB.
        collection_name: Nom de la collection.
        embedding_model_name: Modèle de sentence-transformers à utiliser.
    
    Returns:
        Statistiques d'indexation.
    """
    print(f"🔄 Chargement du modèle d'embedding : {embedding_model_name}")
    model = SentenceTransformer(embedding_model_name)
    
    print(f"🗄️  Connexion à ChromaDB : {chroma_path}")
    client = get_chroma_client(chroma_path)
    collection = get_or_create_collection(client, collection_name)
    
    print(f"📂 Indexation du dossier : {docs_dir}")
    stats = index_directory(Path(docs_dir), collection, model)
    
    return {
        **stats,
        "collection_size": collection.count(),
        "collection_name": collection_name,
    }