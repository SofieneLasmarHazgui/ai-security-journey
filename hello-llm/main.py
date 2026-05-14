"""
Premier appel à l'API Claude.
Auteur: Sofiene Lasmar
Projet: ai-security-journey
"""
import os
from anthropic import Anthropic
from dotenv import load_dotenv

# Charge les variables d'environnement depuis .env
load_dotenv()

# Initialise le client Anthropic
client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def ask_claude(question: str) -> str:
    """Pose une question à Claude et retourne sa réponse."""
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": question}
        ]
    )
    # response.content est une liste de blocs, on récupère le texte
    return response.content[0].text


def main():
    print("🤖 Premier appel à Claude depuis Ubuntu/WSL\n")
    
    question = (
        "Explique en 3 phrases ce qu'est la prompt injection dans les LLM, "
        "du point de vue d'un ingénieur DevSecOps."
    )
    
    print(f"❓ Question : {question}\n")
    print("⏳ Réponse de Claude :\n")
    
    answer = ask_claude(question)
    print(answer)
    print("\n✅ Premier appel API réussi !")


if __name__ == "__main__":
    main()