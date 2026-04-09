import requests
from bs4 import BeautifulSoup
import sqlite3
import time
import re
from urllib.parse import urljoin

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('rslp')

BASE_URL = "https://www.imoveis-sc.com.br/blumenau/comprar?page={}"
HEADERS = {"User-Agent": "Mozilla/5.0"}
DELAY = 1
TOTAL_PAGINAS = 10

conn = sqlite3.connect("imoveis.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS imoveis (
    id INTEGER PRIMARY KEY,
    preco TEXT,
    localizacao TEXT,
    descricao TEXT,
    descricao_limpa TEXT,
    tokens TEXT,
    tokens_sem_stopwords TEXT,
    tokens_stemizados TEXT,
    url TEXT
)
""")

conn.commit()

def limpar_texto(texto):
    if not texto:
        return None

    texto = BeautifulSoup(texto, "html.parser").get_text()

    texto = re.sub(r'http\S+|www\S+', '', texto)

    texto = texto.lower()

    # remover caracteres especiais (mantendo letras e números)
    texto = re.sub(r'[^a-zà-ú0-9\s]', '', texto)

    # normalizar espaços
    texto = re.sub(r'\s+', ' ', texto).strip()

    return texto

stop_words = set(stopwords.words('portuguese'))
stemmer = RSLPStemmer()

def processar_nlp(texto):
    if not texto:
        return None, None, None

    # tokenização
    tokens = word_tokenize(texto)

    # remover stopwords
    tokens_sem_stopwords = [
        t for t in tokens if t not in stop_words
    ]

    # stemming
    tokens_stemizados = [
        stemmer.stem(t) for t in tokens_sem_stopwords
    ]

    return tokens, tokens_sem_stopwords, tokens_stemizados

def extrair_imovel(url):
    try:
        match = re.search(r'-(\d+)\.html', url)
        imovel_id = int(match.group(1)) if match else None

        if not imovel_id:
            return

        response = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(response.text, "html.parser")

        preco = None
        for tag in soup.find_all(["h2", "h3", "strong"]):
            if tag.text and "R$" in tag.text:
                preco = tag.text.strip()
                break

        titulo = soup.find("h1")
        bairro = None

        if titulo:
            texto = titulo.text.strip()
            match = re.search(r'Bairro (.*?) em Blumenau', texto)
            if match:
                bairro = match.group(1)

        descricao = None

        for div in soup.find_all("div"):
            if "Sobre o imóvel" in div.text:
                texto = div.text

                partes = texto.split("Sobre o imóvel")
                descricao = partes[1] if len(partes) > 1 else texto

                lixo_markers = [
                    "Ver mais imóveis",
                    "Fale com o anunciante",
                    "Telefone",
                    "WhatsApp",
                    "Anunciante deste imóvel"
                ]

                for marker in lixo_markers:
                    if marker in descricao:
                        descricao = descricao.split(marker)[0]

                descricao = descricao.strip()
                descricao = re.sub(r'\n+', '\n', descricao)
                break

        descricao_limpa = limpar_texto(descricao)

        tokens, tokens_sem_stopwords, tokens_stemizados = processar_nlp(descricao_limpa)

        tokens_str = " ".join(tokens) if tokens else None
        tokens_sw_str = " ".join(tokens_sem_stopwords) if tokens_sem_stopwords else None
        tokens_stem_str = " ".join(tokens_stemizados) if tokens_stemizados else None

        cursor.execute("""
        INSERT OR IGNORE INTO imoveis (
            id, preco, localizacao, descricao,
            descricao_limpa,
            tokens,
            tokens_sem_stopwords,
            tokens_stemizados,
            url
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            imovel_id,
            preco,
            bairro,
            descricao,
            descricao_limpa,
            tokens_str,
            tokens_sw_str,
            tokens_stem_str,
            url
        ))

        conn.commit()

    except Exception as e:
        print("Erro no imóvel:", url, e)




for pagina in range(1, TOTAL_PAGINAS + 1):
    print(f"\nPágina {pagina}")

    url = BASE_URL.format(pagina)
    response = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(response.text, "html.parser")

    links = []

    for a in soup.find_all("a", href=True):
        href = a["href"]

        if "/blumenau/comprar/" in href and ".html" in href:
            link_completo = urljoin("https://www.imoveis-sc.com.br", href)

            if link_completo not in links:
                links.append(link_completo)

    print(f"{len(links)} imóveis encontrados")

    for link in links:
        print("→", link)
        extrair_imovel(link)
        time.sleep(DELAY)

    time.sleep(DELAY)

conn.close()
print("\nFinalizado!")