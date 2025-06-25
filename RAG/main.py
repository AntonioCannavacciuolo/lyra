import os

# Imposta cache environment (adattare il percorso se necessario)
os.environ["HF_HOME"] = "C:/Users/canna/huggingface_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "C:/Users/canna/huggingface_cache"
os.environ["TRANSFORMERS_CACHE"] = "C:/Users/canna/huggingface_cache"

from FlagEmbedding import BGEM3FlagModel
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
import requests


def load_sentences_from_file(filepath):
    """Carica frasi da un file di testo locale."""
    print(f"Caricamento da file: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"File non trovato: {filepath}")
        return []
    sentences = [s.strip() for s in re.split(r'(?<=[\.\?\!])\s+', content) if s.strip()]
    print(f"Caricate {len(sentences)} frasi dal file.")
    return sentences


def load_sentences_from_wikipedia(title, language='it'):
    """Scarica e segmenta il contenuto di un articolo Wikipedia tramite API."""
    api_url = f"https://{language}.wikipedia.org/w/api.php"
    params = {
        'action': 'query',
        'prop': 'extracts',
        'explaintext': True,
        'titles': title,
        'format': 'json',
        'redirects': 1
    }
    print(f"Richiesta API Wikipedia per: {title}")
    try:
        resp = requests.get(api_url, params=params, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"Errore API Wikipedia: {e}")
        return []
    data = resp.json()
    pages = data.get('query', {}).get('pages', {})
    extract = next(iter(pages.values())).get('extract', '')
    if not extract:
        print(f"Nessun estratto trovato per {title}.")
        return []
    sentences = [s.strip() for s in re.split(r'(?<=[\.\?\!])\s+', extract) if s.strip()]
    print(f"Caricate {len(sentences)} frasi da Wikipedia ({language}) articolo '{title}'.")
    return sentences


def load_sentences_from_github_search(topic, per_page=5, branch='main'):
    """Cerca repository su GitHub per 'topic', scarica README e segmenta in frasi."""
    token = os.getenv('GITHUB_TOKEN')
    headers = {'Accept': 'application/vnd.github.v3+json'}
    if token:
        headers['Authorization'] = f'token {token}'
    search_url = 'https://api.github.com/search/repositories'
    params = {'q': topic, 'per_page': per_page}
    print(f"Ricerca GitHub per repository con topic: '{topic}'")
    try:
        resp = requests.get(search_url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"Errore ricerca GitHub: {e}")
        return []
    items = resp.json().get('items', [])
    all_sentences = []
    for repo in items:
        owner = repo['owner']['login']
        name = repo['name']
        raw_readme = f"https://raw.githubusercontent.com/{owner}/{name}/{branch}/README.md"
        print(f"Scarico README da {owner}/{name}")
        try:
            r = requests.get(raw_readme, timeout=5)
            r.raise_for_status()
            content = r.text
            sentences = [s.strip() for s in re.split(r'(?<=[\.\?\!])\s+', content) if s.strip()]
            all_sentences.extend(sentences)
        except requests.RequestException:
            print(f"README non trovato in {owner}/{name} su branch {branch}.")
            continue
    print(f"Totale frasi caricate da GitHub: {len(all_sentences)}")
    return all_sentences


if __name__ == "__main__":
    # Scegli la sorgente delle frasi: 'file', 'wiki' o 'github'
    source = 'github'

    if source == 'file':
        sentences = load_sentences_from_file("documento.txt")
    elif source == 'wiki':
        sentences = load_sentences_from_wikipedia("Ferrari_California", language='it')
    else:
        # Prima prova GitHub
        sentences = load_sentences_from_github_search("Ferrari California", per_page=5)
        # Fallback su Wikipedia
        if not sentences:
            print("Nessun risultato GitHub, passo a Wikipedia...")
            sentences = load_sentences_from_wikipedia("Ferrari_California", language='it')
        # Ulteriore fallback su file locale
        if not sentences:
            print("Nessun risultato Wikipedia, passo a file locale...")
            sentences = load_sentences_from_file("documento.txt")

    if not sentences:
        print("Nessuna frase caricata da nessuna fonte. Controlla i parametri o il file.")
        exit(1)

    # Inizializza il modello
    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

    # Definisci la tua query di confronto
    query = "la ferrari california quanti cavalli ha"

    # Calcola gli embeddings
    embeddings = np.array(model.encode(sentences)['dense_vecs'])
    embedding_query = np.array(model.encode([query])['dense_vecs'])

    # Combina gli embeddings per la riduzione dimensionale
    all_embeddings = np.vstack([embeddings, embedding_query])

    # Salva gli embeddings su file
    np.save('embeddings.npy', all_embeddings)
    print("Embeddings (inclusa la query) salvati in 'embeddings.npy'")

    # Calcola similarità coseno
    cos_similarities = cosine_similarity(embedding_query, embeddings)[0]

    # Ordina per similarità decrescente
    sorted_indices = np.argsort(cos_similarities)[::-1]
    print("\nFrasi ordinate per similarità con la query:")
    for idx in sorted_indices:
        print(f"{sentences[idx]} --> Similarità: {cos_similarities[idx]:.4f}")

    # Riduzione a 2D con t-SNE
    embedding_2d = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(all_embeddings)

    # Visualizzazione del grafico
    plt.figure(figsize=(10, 8))
    for i, sent in enumerate(sentences):
        x, y = embedding_2d[i]
        plt.scatter(x, y)
        plt.text(x + 0.01, y + 0.01, sent, fontsize=8)

    # Punto della query in rosso
    qx, qy = embedding_2d[-1]
    plt.scatter(qx, qy, marker='x')
    plt.text(qx + 0.01, qy + 0.01, query, fontsize=9, color='red')

    plt.title('t-SNE degli Embeddings')
    plt.xlabel('Componente 1')
    plt.ylabel('Componente 2')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
