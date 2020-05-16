# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 19:35:53 2020

@author: SheilaCarolina
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk import tokenize
import seaborn as sns
from string import punctuation
import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import ngrams

#nltk.download('')

resenha = pd.read_csv("C:\\Users\\SheilaCarolina\\Documents\\NLP\\Curso-P1_PLN\\imdb-reviews-pt-br.csv")
resenha.head()

treino, teste, classe_treino, classe_teste = train_test_split(resenha.text_pt, resenha.sentiment, random_state = 42)

regressao_logistica = LogisticRegression()
#regressao_logistica.fit(treino, classe_treino)
#acuracia = regressao_logistica.score(teste, classe_teste)
#print(acuracia)

#print(resenha.sentiment.value_counts())
#print(resenha.head())

#irá adicionar uma coluna com alguma informação
classificacao = resenha["sentiment"].replace(["neg", "pos"], [0, 1])
print(classificacao)

resenha["classificacao"] = classificacao
print(resenha.head())
print(resenha.tail())

#Bag of Words
#texto = ['Assitir um filme  ótimo', 'Assitir um filme ruim']
#
#vetorizar = CountVectorizer(lowercase=False)#Armazena o dado como ele é escrito realmente
##vetorizar = CountVectorizer()#passa todo mundo pra minusculo
#bag_of_words = vetorizar.fit_transform(texto)
#vetorizar.get_feature_names()
#
##matriz_sparsa = pd.SparseDataFrame(bag_of_words, columns=vetorizar.get_feature_names())#Versão antiga
#matriz_sparsa = pd.DataFrame.sparse.from_spmatrix(bag_of_words, columns=vetorizar.get_feature_names())#Troca o nan por O

def classificar_texto(texto, coluna_texto, coluna_classificacao):
    vetorizar = CountVectorizer(lowercase=False, max_features=50)#vetorizar os dados com 50 dimensoes
    bag_of_words = vetorizar.fit_transform(texto[coluna_texto])#criar a representacao do texto/dataframe e a coluna de interesse no qual o texto está
#    print(bag_of_words.shape)

    treino, teste, classe_treino, classe_teste = train_test_split(bag_of_words, texto[coluna_classificacao], random_state = 42)

    regressao_logistica = LogisticRegression(solver='lbfgs')
    regressao_logistica.fit(treino, classe_treino)
#    acuracia = regressao_logistica.score(teste, classe_teste)
    return(regressao_logistica.score(teste, classe_teste))

print(classificar_texto(resenha, 'text_pt', 'classificacao'))

#matriz_sparsa = pd.SparseDataFrame(bag_of_words, columns=vetorizar.get_feature_names())#Versão antiga
#matriz_sparsa = pd.DataFrame.sparse.from_spmatrix(bag_of_words, columns=vetorizar.get_feature_names())#Troca o nan por O

#WordCloud: vicualização de palavras com mais frequência
todas_palavras = ' '.join([texto for texto in resenha.text_pt])
#nuvem_palavras = WordCloud().generate(todas_palavras)
nuvem_palavras = WordCloud(width = 800, height = 500, background_color = (245, 245, 220), max_font_size = 100, collocations=False).generate(todas_palavras)

#criação da imagem
plt.figure(figsize=(10, 7))
plt.imshow(nuvem_palavras, interpolation='bilinear')
plt.axis("off")
plt.show

#separando os sentimentos positivos dos negativos e vice versa
#resenha.query("sentiment == 'pos'")
def nuvem_negativa(texto, coluna_texto):
    texto_negativo = texto.query("sentiment == 'neg'")
    todas_palavras = ' '.join([texto for texto in texto_negativo[coluna_texto]])
    #nuvem_palavras = WordCloud().generate(todas_palavras)
    nuvem_palavras = WordCloud(width = 800, height = 500, background_color = (117,132,133), max_font_size = 100, collocations=False).generate(todas_palavras)
    
    plt.figure(figsize=(10, 7))
    plt.imshow(nuvem_palavras, interpolation='bilinear')
    plt.axis("off")
    plt.show

def nuvem_positiva(texto, coluna_texto):
    texto_positivo = texto.query("sentiment == 'pos'")
    todas_palavras = ' '.join([texto for texto in texto_positivo[coluna_texto]])
    #nuvem_palavras = WordCloud().generate(todas_palavras)
    nuvem_palavras = WordCloud(width = 800, height = 500, background_color = (254,206,234), max_font_size = 100, collocations=False).generate(todas_palavras)
    
    plt.figure(figsize=(10, 7))
    plt.imshow(nuvem_palavras, interpolation='bilinear')
    plt.axis("off")
    plt.show

nuvem_negativa(resenha, 'text_pt')
nuvem_positiva(resenha, 'text_pt')

#frase = ['um filme ruim', 'um filme bom']
##frequencia = nltk.FreqDist(frase)
##print(frequencia)
#
#frase = "Bem vindo ao mundo do PLN!"
token_espaco = tokenize.WhitespaceTokenizer()
#token_frase = token_espaço.tokenize(frase)
#print(token_frase)

#token_espaco = tokenize.WhitespaceTokenizer()
#token_frase = token_espaco.tokenize(todas_palavras)
#frequencia = nltk.FreqDist(token_frase)

#df_frequencia = pd.DataFrame({'Palavra': list(frequencia.keys()), 'Frequência': list(frequencia.values())})

def pareto(texto, coluna_texto, quantidade):
    todas_palavras = ' '.join([texto for texto in texto[coluna_texto]])

    token_frase = token_espaco.tokenize(todas_palavras)
    frequencia = nltk.FreqDist(token_frase)
    
    df_frequencia = pd.DataFrame({'Palavra': list(frequencia.keys()), 'Frequência': list(frequencia.values())})
    df_frequencia = df_frequencia.nlargest(columns = 'Frequência', n = quantidade)
    
    plt.figure(figsize = (12, 8))
    ax = sns.barplot(data = df_frequencia, x = "Palavra", y = "Frequência", color = 'gray')
    ax.set(ylabel = "Contagem")
    plt.show()

pareto(resenha, "text_pt", 10)#DEMORA UM POUCO CALMA

#Stop Words = Palavras que não agregam em nada
palavras_irrelevantes = nltk.corpus.stopwords.words("portuguese")
print(palavras_irrelevantes)

#retirada das stopwords
#resenha.head()
#DEMORA UM POUCO CALMA
frase_processada = list()
for opniao in resenha.text_pt:
    nova_frase = list()
    palavras_texto = token_espaco.tokenize(opniao)
    for palavra in palavras_texto:
        if palavra not in palavras_irrelevantes:
            nova_frase.append(palavra)
    frase_processada.append(' '.join(nova_frase))

resenha["tratamento1"] = frase_processada

resenha.head()
classificar_texto(resenha, 'tratamento1', 'classificacao')
pareto(resenha, 'tratamento1', 10)

#PARTE 2
#separar a pontuação do texto
frase = "Olá mundo!"
token_pontuacao = tokenize.WordPunctTokenizer()
token_frase = token_pontuacao.tokenize(frase)

print(token_frase)

#retirada de pontuação
print(punctuation)
pontuacao = list()
for ponto in punctuation:
    pontuacao.append(ponto)
print(pontuacao)

pontuacao_stop_words = pontuacao + palavras_irrelevantes

frase_processada = list()
for opiniao in resenha["tratamento1"]:
    nova_frase = list()
    palavras_texto = token_pontuacao.tokenize(opiniao)
    for palavra in palavras_texto:
        if palavra not in pontuacao_stop_words:
            nova_frase.append(palavra)
    frase_processada.append(' '.join(nova_frase))

resenha["tratamento2"] = frase_processada

resenha.head()

print(resenha["tratamento1"][0], '\n\n', resenha["tratamento2"][0])

pareto(resenha, "tratamento2", 10)

#retirada de acentuaçõa
#acentos = "ótimo péssimo não é tão"
#teste = unidecode.unidecode(acentos)
#print(teste)
sem_acentos = [unidecode.unidecode(texto) for texto in resenha["tratamento2"]]
#print(sem_acentos[0])

stopwords_sem_acento = [unidecode.unidecode(texto) for texto in pontuacao_stop_words]

resenha["tratamento3"] = sem_acentos

frase_processada = list()
for opiniao in resenha["tratamento3"]:
    nova_frase = list()
    palavras_texto = token_pontuacao.tokenize(opiniao)
    for palavra in palavras_texto:
        if palavra not in pontuacao_stop_words:
            nova_frase.append(palavra)
    frase_processada.append(' '.join(nova_frase))

resenha["tratamento3"] = frase_processada

resenha.head()

acuracia_tratamento3 = classificar_texto(resenha, "tratamento3", "classificacao")
print(acuracia_tratamento3)

nuvem_negativa(resenha, "tratamento3")
nuvem_positiva(resenha, "tratamento3")
pareto(resenha, "tratamento3", 10)

#frase = "O Thiago é o novo instrutor da Alura"
#print(frase.lower())

frase_processada = list()
for opiniao in resenha["tratamento3"]:
    nova_frase = list()
    opiniao = opiniao.lower()
    palavras_texto = token_pontuacao.tokenize(opiniao)
    for palavra in palavras_texto:
        if palavra not in stopwords_sem_acento:
            nova_frase.append(palavra)
    frase_processada.append(' '.join(nova_frase))

resenha["tratamento4"] = frase_processada

print(resenha["text_pt"][0], '\n\n', resenha["tratamento4"][0])

acuracia_tratamento4 = classificar_texto(resenha, "tratamento4", "classificacao")

print("Tratamento 3 -> ", acuracia_tratamento3, '\nTratamento 4 -> ', acuracia_tratamento4)

nuvem_negativa(resenha, "tratamento4")
nuvem_positiva(resenha, "tratamento4")
pareto(resenha, "tratamento4", 10)

#stemização
stemmer = nltk.RSLPStemmer()
#stemmer.stem("corredor")
#stemmer.stem("corre")
#stemmer.stem("correria")

frase_processada = list()
for opiniao in resenha["tratamento4"]:
    nova_frase = list()
    palavras_texto = token_pontuacao.tokenize(opiniao)
    for palavra in palavras_texto:
#        if palavra not in stopwords_sem_acento:
        nova_frase.append(stemmer.stem(palavra))
    frase_processada.append(' '.join(nova_frase))

resenha["tratamento5"] = frase_processada

print(resenha["text_pt"][0], '\n\n', resenha["tratamento5"][0])

acuracia_tratamento5 = classificar_texto(resenha, "tratamento5", "classificacao")

print("Tratamento 4 -> ", acuracia_tratamento4, '\nTratamento 5 -> ', acuracia_tratamento5)

nuvem_negativa(resenha, "tratamento5")
nuvem_positiva(resenha, "tratamento5")
pareto(resenha, "tratamento5", 10)

#TF-IDF
frase = ['Assisti um filme ótimo', 'Assiti um filme péssimo']
tfidf = TfidfVectorizer(lowercase=False, max_features=50)

caracteristicas = tfidf.fit_transform(frase)
pd.DataFrame(caracteristicas.todense(), columns=tfidf.get_feature_names())

vetor_tfidf_bruto = tfidf.fit_transform(resenha["text_pt"])
treino, teste, classe_treino, classe_teste = train_test_split(vetor_tfidf_bruto, resenha["classificacao"], random_state=42)

regressao_logistica.fit(treino, classe_treino)
acuracia_tfidf_bruto = regressao_logistica.score(teste, classe_teste)
print(acuracia_tfidf_bruto)

vetor_tfidf_tratado = tfidf.fit_transform(resenha["tratamento5"])
treino, teste, classe_treino, classe_teste = train_test_split(vetor_tfidf_tratado, resenha["classificacao"], random_state=42)

regressao_logistica.fit(treino, classe_treino)
acuracia_tfidf_tratado = regressao_logistica.score(teste, classe_teste)
print(acuracia_tfidf_tratado)

#Ngrams
frase = "Assisti um ótimo filme."
frase_separada = token_espaco.tokenize(frase)
pares = ngrams(frase_separada, 2)
list(pares)

tfidf = TfidfVectorizer(lowercase=False, ngram_range = (1, 2))
vetor_tfidf = tfidf.fit_transform(resenha["tratamento5"])
treino, teste, classe_treino, classe_teste = train_test_split(vetor_tfidf, resenha["classificacao"], random_state = 42)
regressao_logistica.fit(treino, classe_treino)
acuracia_tfidf_ngrams = regressao_logistica.score(teste, classe_teste)
print(acuracia_tfidf_ngrams)

#sem ngrams
tfidf = TfidfVectorizer(lowercase=False)
vetor_tfidf = tfidf.fit_transform(resenha["tratamento5"])
treino, teste, classe_treino, classe_teste = train_test_split(vetor_tfidf, resenha["classificacao"], random_state = 42)
regressao_logistica.fit(treino, classe_treino)
acuracia_tfidf = regressao_logistica.score(teste, classe_teste)
print(acuracia_tfidf)

#pesos positivos
pesos = pd.DataFrame(regressao_logistica.coef_[0].T, index = tfidf.get_feature_names())
pesos.nlargest(10, 0)

#pesos negativos
pesos.nsmallest(10, 0)










