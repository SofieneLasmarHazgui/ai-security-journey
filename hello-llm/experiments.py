"""
Expérience 3 : la temperature.
Comment le même prompt peut donner des réponses différentes
selon le paramètre temperature.
"""
import os
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

MODEL = "claude-haiku-4-5-20251001"  # Haiku pour économiser


def ask_with_temperature(question: str, temperature: float, n_times: int = 3):
    """Pose la même question N fois avec une temperature donnée."""
    print(f"\n{'='*70}")
    print(f"🌡️  TEMPERATURE = {temperature}")
    print(f"❓ Question : {question}")
    print(f"{'='*70}")
    
    for i in range(n_times):
        response = client.messages.create(
            model=MODEL,
            max_tokens=100,
            temperature=temperature,
            messages=[{"role": "user", "content": question}]
        )
        print(f"\n--- Essai {i+1} ---")
        print(response.content[0].text)


def main():
    question = "Invente UN seul nom court et original pour une startup de cybersécurité spécialisée IA. Donne juste le nom, rien d'autre."
    
    # Temperature 0 : déterministe
    ask_with_temperature(question, temperature=0.0, n_times=3)
    
    # Temperature 1 : créatif
    ask_with_temperature(question, temperature=1.0, n_times=3)


if __name__ == "__main__":
    main()