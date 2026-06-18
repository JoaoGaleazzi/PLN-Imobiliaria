import sys
sys.dont_write_bytecode = True
import os
import sqlite3
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Configurações
TOP_K = 5
EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
ARQUIVO_EMBEDDINGS = "embeddings.npy"

# Carregar dados
def carregar_dados(db_path="imoveis.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT
        id,
        preco,
        localizacao,
        descricao,
        tokens_sem_stopwords,
        url
        FROM imoveis
        WHERE tokens_sem_stopwords IS NOT NULL
        AND tokens_sem_stopwords != ''
    """)
    linhas = cursor.fetchall()
    conn.close()
    ids = [linha[0] for linha in linhas]
    precos = [linha[1] for linha in linhas]
    localizacoes = [linha[2] for linha in linhas]
    descricoes = [linha[3] for linha in linhas]
    documentos = [linha[4] for linha in linhas]
    urls = [linha[5] for linha in linhas]
    return (
        ids,
        precos,
        localizacoes,
        descricoes,
        documentos,
        urls
    )

# Carregar imóveis
print("\nCarregando imóveis...")
(
    ids,
    precos,
    localizacoes,
    descricoes,
    documentos,
    urls
) = carregar_dados()
print(f"{len(documentos)} imóveis carregados.")

# Embeddings
print("\nCarregando modelo de embeddings...")
modelo_embedding = SentenceTransformer(
    EMBEDDING_MODEL
)
if os.path.exists(ARQUIVO_EMBEDDINGS):
    print("Carregando embeddings salvos...")
    embeddings = np.load(
        ARQUIVO_EMBEDDINGS
    )
else:
    print("Gerando embeddings (primeira execução)...")
    embeddings = modelo_embedding.encode(
        documentos,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    np.save(
        ARQUIVO_EMBEDDINGS,
        embeddings
    )
    print("Embeddings salvos.")

# FAISS
print("\nConstruindo índice vetorial...")
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(
    embeddings.astype("float32")
)
print("Índice criado.")

# Retrieval
def recuperar_imoveis(pergunta, k=TOP_K):
    emb = modelo_embedding.encode(
        [pergunta],
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    scores, indices = index.search(
        emb.astype("float32"), k
    )
    resultados = []
    for score, idx in zip(
        scores[0],
        indices[0]
    ):
        resultados.append({
            "id": ids[idx],
            "preco": precos[idx],
            "localizacao": localizacoes[idx],
            "descricao": descricoes[idx],
            "url": urls[idx],
            "score": float(score)
        })
    return resultados

# Resposta
def gerar_resposta(pergunta, imoveis):
    resposta = []
    resposta.append("RESULTADOS DO RAG")
    resposta.append(f"Busca: {pergunta}\n")
    resposta.append(f"Top {len(imoveis)} imóveis encontrados.\n")
    for i, imovel in enumerate(imoveis):
        resposta.append(f"""
TOP {i+1}
ID:
{imovel['id']}
PREÇO:
{imovel['preco']}
LOCALIZAÇÃO:
{imovel['localizacao']}
SIMILARIDADE:
{imovel['score']:.3f}
DESCRIÇÃO:
{imovel['descricao']}
URL:
{imovel['url']}
""")
    return "\n".join(resposta)

# RAG
def responder(pergunta):
    imoveis = recuperar_imoveis(
        pergunta
    )
    return gerar_resposta(
        pergunta,
        imoveis
    )

# Loop principal
def executar_rag():
    print("\nRAG IMOBILIÁRIO")
    print("\nDigite 'sair' para encerrar.\n")
    while True:
        pergunta = input("\nDigite sua busca: ")
        if pergunta.lower() == "sair":
            print("\nEncerrando...")
            break
        try:
            resposta = responder(
                pergunta
            )
            print(resposta)
        except Exception as e:
            print("\nErro:")
            print(e)

if __name__ == "__main__":
    executar_rag()