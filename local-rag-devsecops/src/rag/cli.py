"""
CLI principale du RAG local DevSecOps.

Commandes :
    rag ingest                    # indexer ./docs/
    rag ask "question"            # poser une question (streaming par défaut)
    rag ask "..." --no-stream     # mode bloquant
    rag info                      # statistiques sur la base
"""
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from rag.ingest import ingest, IngestError
from rag.pipeline import RAGPipeline, RAGError
from rag.retriever import Retriever, RetrieverError


# ---------- Application Typer ----------

app = typer.Typer(
    name="rag",
    help="🛡️  Local RAG for DevSecOps documentation (powered by Ollama).",
    add_completion=False,
)

console = Console()


# ---------- Commande : ingest ----------

@app.command("ingest")
def ingest_cmd(
    docs_dir: Annotated[
        Path,
        typer.Option("--docs", "-d", help="Dossier contenant les documents."),
    ] = Path("./docs"),
    chroma_path: Annotated[
        Path,
        typer.Option("--db", help="Chemin de la vector DB."),
    ] = Path("./vectordb"),
    collection: Annotated[
        str,
        typer.Option("--collection", "-c", help="Nom de la collection."),
    ] = "devsecops_docs",
):
    """
    Indexer les documents dans la vector DB.
    
    Lit les PDF, MD et TXT du dossier, les découpe en chunks,
    génère les embeddings et les stocke dans ChromaDB.
    """
    console.print(Panel(
        f"📂 Documents : [cyan]{docs_dir}[/cyan]\n"
        f"🗄️  Vector DB  : [cyan]{chroma_path}[/cyan]\n"
        f"📛 Collection : [cyan]{collection}[/cyan]",
        title="🔄 Démarrage de l'indexation",
        border_style="cyan",
    ))
    
    try:
        stats = ingest(
            docs_dir=str(docs_dir),
            chroma_path=str(chroma_path),
            collection_name=collection,
        )
    except IngestError as e:
        console.print(f"[red]❌ Erreur d'indexation : {e}[/red]")
        raise typer.Exit(code=1)
    
    # Tableau récapitulatif
    table = Table(title="📊 Statistiques d'indexation", border_style="green")
    table.add_column("Métrique", style="bold")
    table.add_column("Valeur", justify="right")
    table.add_row("Documents traités", str(stats["nb_docs"]))
    table.add_row("Chunks créés", str(stats["nb_chunks"]))
    table.add_row("Documents skippés", str(stats["skipped"]))
    table.add_row("Taille totale collection", str(stats["collection_size"]))
    table.add_row("Nom collection", stats["collection_name"])
    
    console.print(table)


# ---------- Commande : ask ----------

@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="La question à poser.")],
    top_k: Annotated[
        int,
        typer.Option("--top-k", "-k", help="Nombre de chunks à retrouver."),
    ] = 5,
    threshold: Annotated[
        float,
        typer.Option("--threshold", "-t", help="Seuil minimal de similarité (0..1)."),
    ] = 0.3,
    temperature: Annotated[
        float,
        typer.Option("--temperature", help="Créativité du LLM (0..1)."),
    ] = 0.2,
    no_stream: Annotated[
        bool,
        typer.Option("--no-stream", help="Mode bloquant (sans streaming)."),
    ] = False,
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Modèle Ollama à utiliser."),
    ] = "qwen2.5:3b",
    show_sources: Annotated[
        bool,
        typer.Option("--show-sources/--no-sources", help="Afficher les sources."),
    ] = True,
):
    """
    Poser une question à la base documentaire.
    
    Le RAG va :
    1. Chercher les chunks pertinents (recherche sémantique)
    2. Construire un prompt augmenté avec les sources
    3. Générer une réponse via Ollama, en citant les sources
    """
    # Initialisation du pipeline avec spinner
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("🔄 Initialisation du pipeline...", total=None)
            from rag.llm import OllamaClient
            retriever = Retriever()
            llm = OllamaClient(model=model)
            pipeline = RAGPipeline(retriever=retriever, llm_client=llm)
    except (RetrieverError, RAGError, Exception) as e:
        console.print(f"[red]❌ Erreur d'initialisation : {e}[/red]")
        console.print(
            "\n[yellow]Vérifie que :\n"
            "  - Tu as lancé 'rag ingest' avant\n"
            "  - Ollama tourne (curl http://localhost:11434)\n"
            f"  - Le modèle '{model}' est installé (ollama list)[/yellow]"
        )
        raise typer.Exit(code=1)
    
    # Affichage de la question
    console.print(Panel(
        f"[bold]{question}[/bold]\n"
        f"[dim]top_k={top_k} | threshold={threshold} | "
        f"model={model} | temperature={temperature}[/dim]",
        title="❓ Question",
        border_style="cyan",
    ))
    
    # Exécution
    if no_stream:
        _ask_blocking(pipeline, question, top_k, threshold, temperature, show_sources)
    else:
        _ask_streaming(pipeline, question, top_k, threshold, temperature, show_sources)


def _ask_blocking(pipeline, question, top_k, threshold, temperature, show_sources):
    """Mode bloquant : attendre la réponse complète."""
    import time
    
    start = time.time()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("🤖 Génération de la réponse...", total=None)
        try:
            response = pipeline.ask(
                question=question,
                top_k=top_k,
                similarity_threshold=threshold,
                temperature=temperature,
            )
        except RAGError as e:
            console.print(f"[red]❌ Erreur : {e}[/red]")
            raise typer.Exit(code=1)
    elapsed = time.time() - start
    
    border = "yellow" if response.no_context_found else "green"
    console.print(Panel(
        response.answer,
        title=f"💬 Réponse (en {elapsed:.1f}s)",
        border_style=border,
    ))
    
    if show_sources and response.sources:
        _print_sources(response.sources)


def _ask_streaming(pipeline, question, top_k, threshold, temperature, show_sources):
    """Mode streaming : afficher les tokens au fur et à mesure."""
    import time
    
    start = time.time()
    
    # Récupérer le stream et les sources
    try:
        stream, sources = pipeline.ask_stream(
            question=question,
            top_k=top_k,
            similarity_threshold=threshold,
            temperature=temperature,
        )
    except RAGError as e:
        console.print(f"[red]❌ Erreur : {e}[/red]")
        raise typer.Exit(code=1)
    
    # Afficher les sources d'abord (elles sont disponibles avant la génération)
    if show_sources:
        if sources:
            _print_sources(sources, prefix="📚 Sources identifiées :")
        else:
            console.print("[yellow]⚠️  Aucune source pertinente trouvée[/yellow]")
    
    # Stream de la réponse
    console.print("\n[bold green]💬 Réponse :[/bold green]\n")
    
    first_token_time = None
    for token in stream:
        if first_token_time is None:
            first_token_time = time.time() - start
        print(token, end="", flush=True)
    
    elapsed = time.time() - start
    
    # Stats de performance
    console.print(
        f"\n\n[dim]⏱️  1er token : {first_token_time:.1f}s | "
        f"Total : {elapsed:.1f}s[/dim]\n"
    )


def _print_sources(sources, prefix="📚 Sources utilisées :"):
    """Affiche les sources sous forme de liste."""
    console.print(f"\n[bold]{prefix}[/bold]")
    for i, src in enumerate(sources, 1):
        # Code couleur selon la similarité
        if src.similarity > 0.6:
            color = "green"
        elif src.similarity > 0.4:
            color = "yellow"
        else:
            color = "red"
        console.print(
            f"  {i}. [cyan]{src.source}[/cyan] "
            f"(chunk {src.chunk_index + 1}/{src.total_chunks}, "
            f"[{color}]similarity {src.similarity:.2f}[/{color}])"
        )


# ---------- Commande : info ----------

@app.command()
def info(
    chroma_path: Annotated[
        Path,
        typer.Option("--db", help="Chemin de la vector DB."),
    ] = Path("./vectordb"),
    collection: Annotated[
        str,
        typer.Option("--collection", "-c", help="Nom de la collection."),
    ] = "devsecops_docs",
):
    """
    Afficher les statistiques de la base documentaire.
    """
    try:
        retriever = Retriever(
            chroma_path=str(chroma_path),
            collection_name=collection,
        )
    except RetrieverError as e:
        console.print(f"[red]❌ {e}[/red]")
        raise typer.Exit(code=1)
    
    # Statistiques
    total_chunks = retriever.collection_size
    
    # Récupérer un échantillon pour analyser les sources
    sample = retriever._collection.peek(limit=total_chunks)
    sources_count = {}
    for metadata in sample["metadatas"]:
        source = metadata.get("source", "unknown")
        sources_count[source] = sources_count.get(source, 0) + 1
    
    # Tableau récap
    table = Table(title="📊 Base documentaire", border_style="cyan")
    table.add_column("Métrique", style="bold")
    table.add_column("Valeur", justify="right")
    table.add_row("Collection", collection)
    table.add_row("Chemin DB", str(chroma_path))
    table.add_row("Total chunks", str(total_chunks))
    table.add_row("Modèle embedding", retriever.embedding_model_name)
    table.add_row("Documents distincts", str(len(sources_count)))
    
    console.print(table)
    
    # Détail par source
    if sources_count:
        sources_table = Table(title="📚 Détail par document", border_style="green")
        sources_table.add_column("Document", style="cyan")
        sources_table.add_column("Chunks", justify="right")
        for source, count in sorted(sources_count.items(), key=lambda x: -x[1]):
            sources_table.add_row(source, str(count))
        console.print(sources_table)


# ---------- Commande : version ----------

@app.command()
def version():
    """Affiche la version de l'outil."""
    console.print("[bold cyan]local-rag-devsecops[/bold cyan] version [green]0.1.0[/green]")


if __name__ == "__main__":
    app()