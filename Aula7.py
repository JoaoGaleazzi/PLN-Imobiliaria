import sys
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from Aula6 import carregar_corpus, construir_tfidf, imoveis_similares


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

PALAVRAS_ALVO = [
    "apartamento", "casa", "piscina", "jardim", "garagem",
    "centro", "bairro", "quarto", "suíte",
]

CONSULTA_TEXTUAL = "apartamento mobiliado piscina"


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

def _imprimir_secao(titulo):
    print("\n" + "=" * 70)
    print(titulo)
    print("=" * 70)


# ---------------------------------------------------------------------------
# 1. Tokenização
# ---------------------------------------------------------------------------

def tokenizar_documentos(documentos):
    """Tokeniza cada documento por espaço.

    O corpus já vem pré-processado de Aula 5 (tokens_sem_stopwords),
    então basta split() — não precisamos de word_tokenize aqui.
    """
    return [doc.split() for doc in documentos]


# ---------------------------------------------------------------------------
# 2. Word2Vec — treinamento e consultas
# ---------------------------------------------------------------------------

def treinar_word2vec(docs_tok, vector_size=100, window=5, min_count=2):
    """Treina um modelo Word2Vec (CBOW) sobre o corpus tokenizado.

    Parâmetros:
      - vector_size: dimensão dos vetores (100)
      - window: janela de contexto (5 palavras)
      - min_count: ignora palavras com frequência < 2 (filtra hapax)
      - sg=0: usa CBOW (prevê palavra central a partir do contexto)
    """
    return Word2Vec(
        sentences=docs_tok,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        sg=0,
        seed=42,
    )


# ---------------------------------------------------------------------------
# 3. Vetores de documento (média dos word embeddings)
# ---------------------------------------------------------------------------

def vetor_medio_documento(tokens, model_wv, dim):
    """Calcula o vetor de um documento como a média dos vetores das palavras."""
    vetores = [model_wv[t] for t in tokens if t in model_wv]
    if not vetores:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(vetores, axis=0)


def calcular_vetores_documentos(docs_tok, model_wv, dim):
    """Aplica vetor_medio_documento a todos os documentos."""
    return np.vstack(
        [vetor_medio_documento(tokens, model_wv, dim) for tokens in docs_tok]
    )


# ---------------------------------------------------------------------------
# 4. Similaridade e recomendação
# ---------------------------------------------------------------------------

def vizinhos_palavra(model_wv, palavra, n=10):
    """Retorna as n palavras mais similares segundo o Word2Vec."""
    try:
        return model_wv.most_similar(palavra, topn=n)
    except KeyError:
        return []


def imoveis_similares_densos(matriz_vetores, ids, localizacoes, doc_idx, n=5):
    """Encontra os n imóveis mais similares a um imóvel-alvo (por cosseno)."""
    sims = cosine_similarity(
        matriz_vetores[doc_idx].reshape(1, -1), matriz_vetores
    ).ravel()
    sims[doc_idx] = -1.0
    indices = np.argsort(sims)[::-1][:n]
    return [(ids[i], localizacoes[i], float(sims[i])) for i in indices]


def recomendar_por_consulta(consulta, model_wv, matriz_vetores, ids, localizacoes, n=5):
    """Recomenda imóveis a partir de uma consulta textual (Word2Vec)."""
    tokens = consulta.lower().split()
    dim = model_wv.vector_size
    vec = vetor_medio_documento(tokens, model_wv, dim).reshape(1, -1)
    if not np.any(vec):
        return pd.DataFrame(columns=["id", "localizacao", "similaridade"])
    sims = cosine_similarity(vec, matriz_vetores).ravel()
    indices = np.argsort(sims)[::-1][:n]
    return pd.DataFrame({
        "id": [ids[i] for i in indices],
        "localizacao": [localizacoes[i] for i in indices],
        "similaridade": [float(sims[i]) for i in indices],
    })


# ---------------------------------------------------------------------------
# 5. spaCy — vetores pré-treinados
# ---------------------------------------------------------------------------

def calcular_vetores_spacy(documentos, nlp):
    """Calcula o vetor de cada documento usando nlp(doc).vector do spaCy."""
    return np.vstack([nlp(doc).vector for doc in documentos])


def recomendar_por_consulta_spacy(consulta, nlp, matriz_vetores, ids, localizacoes, n=5):
    """Recomenda imóveis a partir de uma consulta textual (spaCy)."""
    vec = nlp(consulta).vector.reshape(1, -1)
    if not np.any(vec):
        return pd.DataFrame(columns=["id", "localizacao", "similaridade"])
    sims = cosine_similarity(vec, matriz_vetores).ravel()
    indices = np.argsort(sims)[::-1][:n]
    return pd.DataFrame({
        "id": [ids[i] for i in indices],
        "localizacao": [localizacoes[i] for i in indices],
        "similaridade": [float(sims[i]) for i in indices],
    })


def vizinhos_spacy_no_vocab(nlp, palavra, vocab_corpus, n=10):
    """Vizinhos semânticos via spaCy, restritos ao vocabulário do corpus.

    Restringir ao vocab do corpus torna a comparação Word2Vec × spaCy
    justa: ambos usam o mesmo conjunto de palavras candidatas.
    """
    lex_alvo = nlp.vocab[palavra]
    if not lex_alvo.has_vector:
        return []
    v_alvo = lex_alvo.vector
    norma_alvo = float(np.linalg.norm(v_alvo))
    if norma_alvo == 0.0:
        return []
    resultados = []
    for termo in vocab_corpus:
        if termo == palavra:
            continue
        lex = nlp.vocab[termo]
        if not lex.has_vector:
            continue
        v = lex.vector
        norma = float(np.linalg.norm(v))
        if norma == 0.0:
            continue
        score = float(np.dot(v_alvo, v) / (norma_alvo * norma))
        resultados.append((termo, score))
    resultados.sort(key=lambda x: x[1], reverse=True)
    return resultados[:n]


# ---------------------------------------------------------------------------
# 6. Clusterização e visualização
# ---------------------------------------------------------------------------

def clusterizar_kmeans(matriz_vetores, k=3, random_state=42):
    """Aplica KMeans aos vetores de documentos."""
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    return km.fit_predict(matriz_vetores)


def plot_heatmap(matriz_vetores, ids, titulo, n_amostra=20):
    """Heatmap de similaridade cosseno entre os primeiros n imóveis."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print(f"[aviso] matplotlib/seaborn nao instalados — pulando '{titulo}'")
        return
    n = min(n_amostra, matriz_vetores.shape[0])
    sim = cosine_similarity(matriz_vetores[:n])
    plt.figure(figsize=(12, 10))
    sns.heatmap(sim, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=ids[:n], yticklabels=ids[:n])
    plt.title(titulo)
    plt.tight_layout()
    plt.show()


def plot_pca_3d(matriz_vetores, documentos, localizacoes, clusters, titulo):
    """Scatter 3D interativo (Plotly) com PCA de 3 componentes."""
    try:
        import plotly.express as px
    except ImportError:
        print(f"[aviso] plotly nao instalado — pulando '{titulo}'")
        return
    coords = PCA(n_components=3).fit_transform(matriz_vetores)
    hover = [f"{loc} — {doc[:80]}" for loc, doc in zip(localizacoes, documentos)]
    fig = px.scatter_3d(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
        color=[str(c) for c in clusters],
        hover_name=hover,
        title=titulo,
        labels={"color": "Cluster"},
    )
    fig.show()


# ===========================================================================
#  MAIN — execução sequencial e didática
# ===========================================================================

if __name__ == "__main__":

    # -----------------------------------------------------------------------
    # 1. Carregar corpus
    # -----------------------------------------------------------------------
    _imprimir_secao("1. Carregando corpus de imoveis.db")
    ids, localizacoes, documentos = carregar_corpus()
    tamanhos = [len(doc.split()) for doc in documentos]
    print(f"Documentos: {len(documentos)}")
    print(f"Tokens por doc (media): {np.mean(tamanhos):.1f}  "
          f"min: {min(tamanhos)}  max: {max(tamanhos)}")

    # -----------------------------------------------------------------------
    # 2. Tokenizar
    # -----------------------------------------------------------------------
    _imprimir_secao("2. Tokenizando documentos")
    docs_tok = tokenizar_documentos(documentos)
    print(f"Exemplo (id={ids[0]}): {docs_tok[0][:12]} ...")

    # -----------------------------------------------------------------------
    # 3. Treinar Word2Vec
    # -----------------------------------------------------------------------
    _imprimir_secao("3. Treinando Word2Vec (CBOW, 100 dims)")
    model = treinar_word2vec(docs_tok)
    vocab = list(model.wv.key_to_index.keys())
    print(f"Vocabulario: {len(vocab)} palavras  |  Dimensao: {model.wv.vector_size}")

    # -----------------------------------------------------------------------
    # 4. Vizinhos semânticos (Word2Vec)
    # -----------------------------------------------------------------------
    _imprimir_secao("4. Vizinhos semanticos — Word2Vec")
    print("O Word2Vec aprende que palavras co-ocorrem em contextos parecidos.")
    print()
    print("NOTA SOBRE O CORPUS PEQUENO:")
    print("  Com apenas 180 anuncios de dominio muito homogeneo, o Word2Vec")
    print("  nao consegue diferenciar bem as palavras — quase todos os scores")
    print("  ficam acima de 0.999. Isso e ESPERADO e sera contrastado com os")
    print("  vetores pre-treinados do spaCy mais adiante.")
    for palavra in PALAVRAS_ALVO:
        vizinhos = vizinhos_palavra(model.wv, palavra, n=5)
        if not vizinhos:
            print(f"\n  '{palavra}' — fora do vocabulario")
            continue
        df = pd.DataFrame(vizinhos, columns=["palavra", "similaridade"])
        print(f"\n  '{palavra}'")
        print(df.to_string(index=False))

    # -----------------------------------------------------------------------
    # 5. Vetores médios por documento (Word2Vec)
    # -----------------------------------------------------------------------
    _imprimir_secao("5. Vetores medios por documento (Word2Vec)")
    dim_w2v = model.wv.vector_size
    vetores_w2v = calcular_vetores_documentos(docs_tok, model.wv, dim_w2v)
    sim_w2v = cosine_similarity(vetores_w2v)
    mask = ~np.eye(sim_w2v.shape[0], dtype=bool)
    print(f"Shape: {vetores_w2v.shape}")
    print(f"Similaridade media (fora da diagonal): {sim_w2v[mask].mean():.4f}")
    print("  (valor proximo de 1.0 confirma o colapso — todos os vetores")
    print("   apontam quase na mesma direcao no espaco de 100 dimensoes)")
    plot_heatmap(vetores_w2v, ids, "Word2Vec — similaridade entre 20 imoveis")

    # -----------------------------------------------------------------------
    # 6. Imóveis similares + recomendação (Word2Vec)
    # -----------------------------------------------------------------------
    _imprimir_secao("6. Imoveis similares e recomendacao (Word2Vec)")
    alvo_idx = 0
    print(f"Alvo: id={ids[alvo_idx]}  loc={localizacoes[alvo_idx]}\n")

    print("Top 5 similares:")
    for imovel_id, loc, score in imoveis_similares_densos(
        vetores_w2v, ids, localizacoes, alvo_idx, n=5
    ):
        print(f"  id={imovel_id:<10} loc={loc!s:<30} sim={score:.4f}")

    print(f"\nRecomendacao para consulta: '{CONSULTA_TEXTUAL}'")
    print(recomendar_por_consulta(
        CONSULTA_TEXTUAL, model.wv, vetores_w2v, ids, localizacoes, n=5
    ).to_string(index=False))

    # -----------------------------------------------------------------------
    # 7. KMeans + PCA 3D (Word2Vec)
    # -----------------------------------------------------------------------
    _imprimir_secao("7. KMeans (k=3) + PCA 3D (Word2Vec)")
    clusters_w2v = clusterizar_kmeans(vetores_w2v, k=3)
    df_clusters = pd.DataFrame({
        "id": ids[:10], "localizacao": localizacoes[:10],
        "cluster": clusters_w2v[:10],
    })
    print(df_clusters.to_string(index=False))
    print("  (com vetores colapsados, os clusters particionam ruido numerico)")
    plot_pca_3d(vetores_w2v, documentos, localizacoes, clusters_w2v,
                "Word2Vec — PCA 3D (KMeans)")

    # -----------------------------------------------------------------------
    # 8. Transição: por que vetores pré-treinados?
    # -----------------------------------------------------------------------
    _imprimir_secao("8. Transicao para vetores pre-treinados")
    print("Ate aqui, o Word2Vec aprendeu APENAS dos 180 anuncios de Blumenau.")
    print("Resultado: vocabulario restrito, vetores saturados (tudo ~1.0).")
    print()
    print("Agora usaremos vetores PRE-TREINADOS do spaCy (pt_core_news_lg),")
    print("aprendidos sobre bilhoes de palavras em portugues.")
    print("Esperamos vizinhanca mais semantica, mas menos especifica do dominio.")

    # -----------------------------------------------------------------------
    # 9. Carregar spaCy
    # -----------------------------------------------------------------------
    _imprimir_secao("9. Carregando spaCy pt_core_news_lg")
    try:
        import spacy
        nlp = spacy.load("pt_core_news_lg")
    except OSError:
        print("[ERRO] Modelo 'pt_core_news_lg' nao instalado.")
        print("Rode: python -m spacy download pt_core_news_lg")
        raise SystemExit(1)
    print(f"Modelo carregado. Vetores: {nlp.vocab.vectors.shape}")

    # -----------------------------------------------------------------------
    # 10. Exemplo didático fora do domínio (spaCy)
    # -----------------------------------------------------------------------
    _imprimir_secao("10. Sanidade dos vetores pre-treinados (spaCy)")
    print("Similaridade entre pares de palavras do dia-a-dia:\n")
    pares = [("gato", "sofá"), ("felino", "gato"),
             ("apartamento", "casa"), ("piscina", "churrasqueira")]
    rows = []
    for a, b in pares:
        va, vb = nlp(a).vector, nlp(b).vector
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        sim = float(np.dot(va, vb) / (na * nb)) if na and nb else 0.0
        rows.append({"palavra_a": a, "palavra_b": b, "similaridade": round(sim, 4)})
    print(pd.DataFrame(rows).to_string(index=False))

    # -----------------------------------------------------------------------
    # 11. Vizinhos semânticos (spaCy) — restritos ao vocab do corpus
    # -----------------------------------------------------------------------
    _imprimir_secao("11. Vizinhos semanticos — spaCy (vocab do corpus)")
    print("Restringir ao vocab do corpus torna a comparacao W2V x spaCy justa.\n")
    vocab_corpus = set(vocab)
    for palavra in PALAVRAS_ALVO:
        vizinhos = vizinhos_spacy_no_vocab(nlp, palavra, vocab_corpus, n=5)
        if not vizinhos:
            print(f"  '{palavra}' — sem vetor")
            continue
        df = pd.DataFrame(vizinhos, columns=["palavra", "similaridade"])
        print(f"  '{palavra}'")
        print(df.to_string(index=False))
        print()

    # -----------------------------------------------------------------------
    # 12. Comparação lado a lado: Word2Vec vs spaCy
    # -----------------------------------------------------------------------
    _imprimir_secao("12. Comparacao lado a lado: Word2Vec vs spaCy")
    print("Coracao pedagogico da aula: dominio-especifico vs. pre-treinado.\n")
    for palavra in PALAVRAS_ALVO:
        viz_w2v = vizinhos_palavra(model.wv, palavra, n=5)
        viz_sp = vizinhos_spacy_no_vocab(nlp, palavra, vocab_corpus, n=5)
        rows = []
        for i in range(5):
            w = viz_w2v[i][0] if i < len(viz_w2v) else "-"
            s = viz_sp[i][0] if i < len(viz_sp) else "-"
            rows.append({"rank": i + 1, "Word2Vec": w, "spaCy": s})
        print(f"Palavra: {palavra}")
        print(pd.DataFrame(rows).to_string(index=False))
        print()

    # -----------------------------------------------------------------------
    # 13. Vetores de documento + heatmap + clusters (spaCy)
    # -----------------------------------------------------------------------
    _imprimir_secao("13. Vetores de documento via spaCy + KMeans")
    vetores_spacy = calcular_vetores_spacy(documentos, nlp)
    sim_spacy = cosine_similarity(vetores_spacy)
    mask_s = ~np.eye(sim_spacy.shape[0], dtype=bool)
    print(f"Shape: {vetores_spacy.shape}")
    print(f"Similaridade media (fora da diagonal): {sim_spacy[mask_s].mean():.4f}")
    print("  (0.89 — alto por ser dominio homogeneo, mas diferencia imoveis)")

    plot_heatmap(vetores_spacy, ids,
                 "spaCy pt_core_news_lg — similaridade entre 20 imoveis")

    clusters_spacy = clusterizar_kmeans(vetores_spacy, k=3)
    plot_pca_3d(vetores_spacy, documentos, localizacoes, clusters_spacy,
                "spaCy — PCA 3D (KMeans)")

    # -----------------------------------------------------------------------
    # 14. Imóveis similares + recomendação (spaCy)
    # -----------------------------------------------------------------------
    _imprimir_secao("14. Imoveis similares e recomendacao (spaCy)")
    print(f"Alvo: id={ids[alvo_idx]}  loc={localizacoes[alvo_idx]}\n")

    print("Top 5 similares:")
    for imovel_id, loc, score in imoveis_similares_densos(
        vetores_spacy, ids, localizacoes, alvo_idx, n=5
    ):
        print(f"  id={imovel_id:<10} loc={loc!s:<30} sim={score:.4f}")

    print(f"\nRecomendacao para consulta: '{CONSULTA_TEXTUAL}'")
    print(recomendar_por_consulta_spacy(
        CONSULTA_TEXTUAL, nlp, vetores_spacy, ids, localizacoes, n=5
    ).to_string(index=False))

    # -----------------------------------------------------------------------
    # 15. Comparação tripla final: TF-IDF vs Word2Vec vs spaCy
    # -----------------------------------------------------------------------
    _imprimir_secao("15. Comparacao tripla: TF-IDF vs Word2Vec vs spaCy")
    matriz_tfidf, _ = construir_tfidf(documentos)
    tfidf_sim = imoveis_similares(matriz_tfidf, ids, localizacoes, alvo_idx, n=5)
    w2v_sim = imoveis_similares_densos(vetores_w2v, ids, localizacoes, alvo_idx, n=5)
    spacy_sim = imoveis_similares_densos(vetores_spacy, ids, localizacoes, alvo_idx, n=5)

    print(f"Alvo: id={ids[alvo_idx]}  loc={localizacoes[alvo_idx]}\n")
    rows = []
    for i in range(5):
        rows.append({
            "rank": i + 1,
            "TF-IDF": f"id={tfidf_sim[i][0]} {tfidf_sim[i][1]} ({tfidf_sim[i][2]:.3f})",
            "Word2Vec": f"id={w2v_sim[i][0]} {w2v_sim[i][1]} ({w2v_sim[i][2]:.3f})",
            "spaCy": f"id={spacy_sim[i][0]} {spacy_sim[i][1]} ({spacy_sim[i][2]:.3f})",
        })
    print(pd.DataFrame(rows).to_string(index=False))

    print()
    print("Leitura dos resultados:")
    print("  - TF-IDF: premia overlap lexico (palavras em comum).")
    print("  - Word2Vec: com corpus de 20.000 ele se assemelha com o spaCy.")
    print("  - spaCy: vetores pre-treinados generalizam bem, diferenciam")
    print("    imoveis mesmo em dominio homogeneo (scores 0.80–0.94).")

    print("\nFinalizado!")
