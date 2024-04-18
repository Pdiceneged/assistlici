import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import base64

st.set_page_config(
    page_title="Assistente Contratos",
    page_icon="👍"
)

@st.cache_data()
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_img_as_base64("pdipaper5.png")
img2 = get_img_as_base64("pdiside.png")

page_bg_img = f"""
<style>
header, footer {{
    visibility: hidden !important;
}}

#MainMenu {{
    visibility: visible !important;
    color: #F44D00;
}}

[data-testid="stAppViewContainer"] > .main {{
    background-image: url("data:fundoesg4k/png;base64,{img}");
    background-size: cover; 
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}
[data-testid="stSidebar"] > div:first-child {{
    background-image: url("data:esgfundo1/png;base64,{img2}");
    background-position: center; 
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}

[data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
    right: 2rem;
}}

.stTextInput>div>div>input[type="text"] {{
    background-color: #C5D6ED; 
    color: #000; 
    border-radius: 7px; 
    border: 2px solid #000010; 
    padding: 5px; 
    width: 500;
}}

@media (max-width: 360px) {{
    [data-testid="stAppViewContainer"] > .main, [data-testid="stSidebar"] > div:first-child {{
        background-size: auto;
    }}
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)
st.sidebar.image("Logopdi.png", width=250)

def carregar_dados(caminho):
    return pd.read_excel(caminho)

def encontrar_frase_similar(frase, dados, coluna_comparar, coluna_retornar):
    if frase:
        dados[coluna_comparar] = dados[coluna_comparar].fillna("")
        tfidf_vectorizer = TfidfVectorizer()
        todas_frases = dados[coluna_comparar].tolist() + [frase]
        tfidf_matrix = tfidf_vectorizer.fit_transform(todas_frases)
        cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
        indices_maximos = cosine_sim.argsort()[-3:][::-1]

        resultados = []
        resultado_unicos = set()
        for indice in indices_maximos:
            obj_sucinto = dados.iloc[indice][coluna_retornar]
            if obj_sucinto not in resultado_unicos:
                resultados.append((obj_sucinto, cosine_sim[indice]))
                resultado_unicos.add(obj_sucinto)
            if len(resultados) == 3:
                break
        return resultados
    return []

st.title('Assistente de Preenchimento (Licitações)')

dados = carregar_dados('todas.xlsx')

frase_usuario = st.text_input('Digite o Objeto de Contrato:', '')

if st.button('Analisar'):
    resultados_similares = encontrar_frase_similar(frase_usuario, dados, 'Objeto do contrato', 'Objeto Sucinto')
    if resultados_similares:
        mensagens = ["A opção de Objeto Sucinto mais recomendada é:", "A segunda opção de Objeto Sucinto é esta:", "A terceira opção de Objeto Sucinto é essa:"]
        for i, (natureza_similar, similaridade) in enumerate(resultados_similares):
            st.write(f"**{mensagens[i]}** {natureza_similar}")
            st.write(f"**Grau de Similaridade:** {similaridade * 100:.2f}%")
            st.markdown("---")
    else:
        st.write("Digite uma frase para comparar.")

texto_explicativo = st.write("A primeira opção no resultado é mais recomendada para ser preenchida como Objeto Sucinto, porém, há outras duas opções caso você veja que outra se encaixa melhor.")

st.sidebar.markdown("---")
st.sidebar.markdown("Desenvolvido por [PedroFS](https://linktr.ee/Pedrofsf)")