
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Análise Mega-Sena", layout="wide")
st.title("🎯 Análise e Previsão da Mega-Sena")

# Upload do arquivo
uploaded_file = st.file_uploader("Envie o arquivo .xlsx da Mega-Sena", type="xlsx")

if uploaded_file:
    df = pd.read_excel(uploaded_file, skiprows=5)
    df.columns = df.iloc[0]
    df = df[1:].reset_index(drop=True)
    cols = [f'bola {i}' for i in range(1, 7)]

    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    draws = df[cols].dropna().astype(int).values

    st.subheader("📊 Frequência das Dezenas")
    all_numbers = np.concatenate(draws)
    frequencies = pd.Series(all_numbers).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(12, 4))
    frequencies.plot(kind='bar', ax=ax)
    ax.set_title('Frequência dos Números Sorteados')
    ax.set_xlabel('Dezena')
    ax.set_ylabel('Frequência')
    st.pyplot(fig)

    st.subheader("🧠 Previsão de Novo Jogo")

    def encode_draw(draw):
        out = np.zeros(60)
        for n in draw:
            out[n-1] = 1
        return out

    X = np.array([encode_draw(draws[i]) for i in range(len(draws)-1)])
    y = np.array([encode_draw(draws[i+1]) for i in range(len(draws)-1)])

    models = []
    for i in range(60):
        model = DecisionTreeClassifier(max_depth=10)
        model.fit(X, y[:, i])
        models.append(model)

    last_draw = encode_draw(draws[-1]).reshape(1, -1)
    probs = np.array([model.predict(last_draw)[0] for model in models])
    nums = probs.argsort()[-6:][::-1] + 1

    previous = set(tuple(sorted(x)) for x in draws)
    while tuple(sorted(nums)) in previous:
        probs[np.argmax(probs)] = 0
        nums = probs.argsort()[-6:][::-1] + 1

    st.success(f"🔮 Jogo sugerido que nunca saiu: {sorted(nums)}")

else:
    st.info("Faça o upload de um arquivo .xlsx da Mega-Sena para iniciar a análise.")
