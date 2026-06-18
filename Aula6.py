import sys
sys.dont_write_bytecode = True
import sqlite3

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


TOKEN_PATTERN = r"(?u)\b\w+\b"


def carregar_corpus(db_path="imoveis.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, localizacao, tokens_sem_stopwords
        FROM imoveis
        WHERE tokens_sem_stopwords IS NOT NULL
          AND tokens_sem_stopwords != ''
    """)
    linhas = cursor.fetchall()
    conn.close()

    ids = [linha[0] for linha in linhas]
    localizacoes = [linha[1] for linha in linhas]
    documentos = [linha[2] for linha in linhas]

    return ids, localizacoes, documentos


def construir_bow(documentos):
    vectorizer = CountVectorizer(token_pattern=TOKEN_PATTERN)
    matriz = vectorizer.fit_transform(documentos)
    return matriz, vectorizer


def construir_tfidf(documentos):
    vectorizer = TfidfVectorizer(token_pattern=TOKEN_PATTERN)
    matriz = vectorizer.fit_transform(documentos)
    return matriz, vectorizer


def top_termos_frequentes(matriz_bow, vectorizer, n=20):
    frequencias = np.asarray(matriz_bow.sum(axis=0)).ravel()
    vocab = vectorizer.get_feature_names_out()
    indices = np.argsort(frequencias)[::-1][:n]
    return [(vocab[i], int(frequencias[i])) for i in indices]


def top_termos_distintivos(vectorizer_tfidf, n=20):
    vocab = vectorizer_tfidf.get_feature_names_out()
    idf = vectorizer_tfidf.idf_
    indices = np.argsort(idf)[::-1][:n]
    return [(vocab[i], float(idf[i])) for i in indices]


def top_tfidf_por_documento(matriz_tfidf, vectorizer, doc_idx, n=10):
    vocab = vectorizer.get_feature_names_out()
    linha = matriz_tfidf[doc_idx].toarray().ravel()
    indices = np.argsort(linha)[::-1][:n]
    return [(vocab[i], float(linha[i])) for i in indices if linha[i] > 0]


def imoveis_similares(matriz_tfidf, ids, localizacoes, doc_idx, n=5):
    similaridades = cosine_similarity(matriz_tfidf[doc_idx], matriz_tfidf).ravel()
    similaridades[doc_idx] = -1
    indices = np.argsort(similaridades)[::-1][:n]
    return [
        (ids[i], localizacoes[i], float(similaridades[i]))
        for i in indices
    ]


def _imprimir_secao(titulo):
    print("\n" + "=" * 70)
    print(titulo)
    print("=" * 70)


if __name__ == "__main__":
    _imprimir_secao("Carregando corpus de imoveis.db")
    ids, localizacoes, documentos = carregar_corpus()
    tamanhos = [len(doc.split()) for doc in documentos]
    print(f"Documentos carregados: {len(documentos)}")
    print(f"Tamanho medio (tokens): {np.mean(tamanhos):.1f}")
    print(f"Tamanho min/max: {min(tamanhos)} / {max(tamanhos)}")

    _imprimir_secao("Construindo Bag-of-Words")
    matriz_bow, vec_bow = construir_bow(documentos)
    print(f"Shape da matriz BoW: {matriz_bow.shape}")
    print(f"Tamanho do vocabulario: {len(vec_bow.get_feature_names_out())}")
    print(f"Densidade: {matriz_bow.nnz / (matriz_bow.shape[0] * matriz_bow.shape[1]):.4%}")

    _imprimir_secao("Construindo TF-IDF")
    matriz_tfidf, vec_tfidf = construir_tfidf(documentos)
    print(f"Shape da matriz TF-IDF: {matriz_tfidf.shape}")

    _imprimir_secao("Top 20 termos mais frequentes (BoW)")
    print("Palavras 'obvias' do dominio imobiliario — aparecem em quase todo anuncio")
    for termo, freq in top_termos_frequentes(matriz_bow, vec_bow, n=20):
        print(f"  {termo:<25} {freq}")

    _imprimir_secao("Top 20 termos mais distintivos (maior IDF)")
    print("Palavras raras — aparecem em poucos anuncios, logo discriminam mais")
    for termo, idf in top_termos_distintivos(vec_tfidf, n=20):
        print(f"  {termo:<25} IDF={idf:.3f}")

    _imprimir_secao("Top TF-IDF por documento (3 exemplos)")
    exemplos = [0, len(documentos) // 2, len(documentos) - 1]
    for doc_idx in exemplos:
        print(f"\nImovel id={ids[doc_idx]}  localizacao={localizacoes[doc_idx]}")
        for termo, peso in top_tfidf_por_documento(matriz_tfidf, vec_tfidf, doc_idx, n=10):
            print(f"  {termo:<25} {peso:.4f}")

    _imprimir_secao("Imoveis similares por cosseno TF-IDF")
    alvo_idx = 0
    print(f"Alvo: id={ids[alvo_idx]}  localizacao={localizacoes[alvo_idx]}")
    print("\nTop 5 mais similares:")
    for imovel_id, loc, score in imoveis_similares(matriz_tfidf, ids, localizacoes, alvo_idx, n=5):
        print(f"  id={imovel_id:<10} loc={loc!s:<30} similaridade={score:.4f}")

    print("\nFinalizado!")
