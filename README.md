# 🎯 FaceSense – Reconhecimento de Microgestos e Fadiga  

O **FaceSense** é um projeto de pesquisa (TCC) que desenvolve um sistema inteligente para **detecção de sinais sutis de cansaço, ansiedade e inquietação** em ambientes de estudo e trabalho.  

Ele utiliza técnicas de **visão computacional**, **aprendizado de máquina** e **interface humano-computador** para analisar microgestos, posturas e expressões faciais em tempo real, oferecendo **feedback adaptativo** ao usuário para melhorar sua **produtividade, bem-estar e prevenção de fadiga**.  

---

## 🔎 Motivação  

Com a expansão do trabalho e estudo remoto, aumentaram os desafios relacionados à:  

- **Fadiga digital** (uso prolongado do computador);  
- **Ansiedade e inquietação** durante longos períodos de concentração;  
- **Problemas de postura e saúde mental** em home office;  

Poucos sistemas existentes conseguem analisar **microgestos e sinais sutis** de comportamento de forma contínua e personalizada. O **FaceSense** busca preencher essa lacuna, trazendo uma ferramenta de apoio para **autorregulação e prevenção de burnout**.  

---

## 🚀 Objetivos  

- Detectar **microgestos faciais e corporais** (ex.: toques no rosto, movimentos repetitivos).  
- Identificar **expressões de cansaço e fadiga** a partir de sequências temporais.  
- Fornecer **feedback em tempo real** através de interface interativa.  
- Apoiar usuários em manter **postura, foco e intervalos estratégicos**.  

---

## 🧩 Tecnologias Utilizadas  

- **Python 3.9+**  
- **TensorFlow / Keras** → Modelagem com redes neurais LSTM  
- **Scikit-learn** → Suporte ao pré-processamento  
- **NumPy / Pandas** → Manipulação dos dados  
- **OpenCV + MediaPipe** → Captura de vídeo e landmarks em tempo real  
- **Matplotlib / Seaborn** → Visualização de métricas e análises
- **FastAPI**
- **Streamlit**

---

## 📂 Base de Dados  

O projeto utiliza o **iMiGUE Dataset** (University of Oulu), composto por:  
- **Skeleton data** (juntas corporais, mãos e rosto – 411 pontos por frame);  
- **Vídeos RGB**;  
- **Labels anotados** com categorias de comportamento.  

---

## 📊 Como o sistema funciona  

1. **Coleta de dados** (skeleton e vídeos RGB).  
2. **Pré-processamento**: normalização e padronização das sequências.  
3. **Treinamento** de modelos LSTM com dados processados.  
4. **Avaliação** no conjunto de teste com métricas (acurácia, F1-score, etc.).  
5. **Inferência em tempo real** com feedback adaptativo ao usuário.  

---

## 📑 Situação Atual  

- ✅ Estruturação dos dados (skeleton → `.npy`)  
- ✅ Definição da arquitetura inicial (LSTM)  
- 🔄 Implementação e avaliação do modelo  
- 🔜 Desenvolvimento da interface interativa (feedback em tempo real)  

---

## ✨ Impacto Esperado  

O **FaceSense** busca contribuir para:  

- Monitorar **produtividade e saúde** em home office;  
- Evitar **burnout** por excesso de exposição à tela;  
- Melhorar **consciência corporal e emocional**;  
- Incentivar o **descanso estratégico**.  
