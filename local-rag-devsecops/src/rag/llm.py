"""
Module d'interface avec Ollama.

Ollama expose une API REST sur http://localhost:11434.
Ce module abstrait les appels et expose 2 modes : 
- generate() : récupère la réponse complète (utile pour tests/batch)
- stream() : itère sur les tokens au fur et à mesure (utile pour CLI/UI)
"""
import json
from dataclasses import dataclass
from typing import Iterator, Optional

import requests


# Configuration par défaut
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:3b"
DEFAULT_TIMEOUT = 120  # 2 minutes — inférence CPU peut être lente
DEFAULT_TEMPERATURE = 0.2  # Bas pour cohérence (réflexe sécurité)


@dataclass
class LLMResponse:
    """Réponse complète d'un appel LLM."""
    content: str
    model: str
    total_duration_ms: int    # temps total en millisecondes
    prompt_tokens: int        # tokens dans le prompt envoyé
    response_tokens: int      # tokens générés


class LLMError(Exception):
    """Erreur durant l'appel au LLM."""
    pass


class OllamaClient:
    """
    Client pour l'API Ollama locale.
    
    Pattern proche du SDK Anthropic : on instancie une fois,
    on appelle generate() ou stream() autant qu'on veut.
    """
    
    def __init__(
        self,
        host: str = DEFAULT_OLLAMA_HOST,
        model: str = DEFAULT_MODEL,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        """
        Initialise le client.
        
        Args:
            host: URL du serveur Ollama (ex: http://localhost:11434).
            model: Nom du modèle (doit avoir été pull via 'ollama pull').
            timeout: Timeout HTTP en secondes.
        """
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        
        # Vérification rapide que Ollama répond
        self._check_health()
    
    def _check_health(self) -> None:
        """Vérifie qu'Ollama est joignable."""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            response.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise LLMError(
                f"Impossible de joindre Ollama sur {self.host}. "
                f"Vérifie qu'il tourne : 'curl {self.host}'"
            )
        except Exception as e:
            raise LLMError(f"Health check Ollama échoué : {e}")
        
        # Vérifier que le modèle est disponible
        available_models = [m["name"] for m in response.json().get("models", [])]
        if self.model not in available_models:
            raise LLMError(
                f"Modèle '{self.model}' introuvable. "
                f"Modèles disponibles : {available_models}. "
                f"Télécharge avec : 'ollama pull {self.model}'"
            )
    
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Génère une réponse complète (bloquant jusqu'à la fin).
        
        Args:
            prompt: Le message utilisateur.
            system: Instructions système optionnelles (rôle du modèle).
            temperature: 0 = déterministe, 1 = créatif. Défaut bas pour sécurité.
            max_tokens: Limite de tokens en sortie (None = pas de limite).
        
        Returns:
            LLMResponse avec contenu et métriques.
        """
        payload = self._build_payload(
            prompt, system, temperature, max_tokens, stream=False
        )
        
        try:
            response = requests.post(
                f"{self.host}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.exceptions.Timeout:
            raise LLMError(
                f"Timeout après {self.timeout}s. "
                f"Augmente le timeout ou utilise un modèle plus petit."
            )
        except requests.exceptions.RequestException as e:
            raise LLMError(f"Erreur HTTP vers Ollama : {e}")
        
        data = response.json()
        
        # Validation de la réponse
        if "message" not in data or "content" not in data["message"]:
            raise LLMError(f"Format de réponse Ollama inattendu : {data}")
        
        return LLMResponse(
            content=data["message"]["content"],
            model=data.get("model", self.model),
            total_duration_ms=data.get("total_duration", 0) // 1_000_000,
            prompt_tokens=data.get("prompt_eval_count", 0),
            response_tokens=data.get("eval_count", 0),
        )
    
    def stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        """
        Génère une réponse en streaming (yield token par token).
        
        Args:
            prompt: Le message utilisateur.
            system: Instructions système.
            temperature: Voir generate().
            max_tokens: Voir generate().
        
        Yields:
            Chaque morceau de texte au fur et à mesure de la génération.
        """
        payload = self._build_payload(
            prompt, system, temperature, max_tokens, stream=True
        )
        
        try:
            with requests.post(
                f"{self.host}/api/chat",
                json=payload,
                timeout=self.timeout,
                stream=True,
            ) as response:
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    
                    # Chaque chunk contient une partie du message
                    if "message" in chunk and "content" in chunk["message"]:
                        token = chunk["message"]["content"]
                        if token:
                            yield token
                    
                    # Fin du stream
                    if chunk.get("done"):
                        break
        
        except requests.exceptions.Timeout:
            raise LLMError(f"Timeout streaming après {self.timeout}s")
        except requests.exceptions.RequestException as e:
            raise LLMError(f"Erreur HTTP streaming : {e}")
    
    def _build_payload(
        self,
        prompt: str,
        system: Optional[str],
        temperature: float,
        max_tokens: Optional[int],
        stream: bool,
    ) -> dict:
        """Construit le payload JSON pour l'API Ollama."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        options = {"temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        
        return {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": options,
        }