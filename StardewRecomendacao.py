import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from unidecode import unidecode
import re

def normaliza_texto(texto):
    if not isinstance(texto, str):
        return ""
    texto = unidecode(texto.lower())
    texto = re.sub(r'[^\w\s]', '', texto)
    return texto.strip()

def processa_lista_presentes(lista_str):
    if pd.isna(lista_str) or lista_str == '[]':
        return []
    
    try:
        lista = ast.literal_eval(lista_str)
        if isinstance(lista, list):
            return lista
        
        lista_str = lista_str.strip("[]").replace("'", "").replace('"', '')
        return [item.strip() for item in lista_str.split(",") if item.strip()]
    except:
        return []

def carrega_dados():
    try:
        dados = pd.read_csv('characters.csv')
        
        colunas_listas = ['Loved Gifts', 'Liked Gifts', 'Neutral Gifts', 
                         'Disliked Gifts', 'Hated Gifts', 'Family']
        
        for coluna in colunas_listas:
            dados[coluna] = dados[coluna].apply(processa_lista_presentes)
        
        for categoria in ['Loved Gifts', 'Liked Gifts', 'Neutral Gifts', 'Disliked Gifts', 'Hated Gifts']:
            dados[categoria+'_normalizado'] = dados[categoria].apply(
                lambda x: [normaliza_texto(item) for item in x]
            )
        
        dados['todos_presentes'] = dados.apply(
            lambda linha: ' '.join(
                linha['Loved Gifts'] + linha['Liked Gifts'] + 
                linha['Neutral Gifts'] + linha['Disliked Gifts'] + 
                linha['Hated Gifts']
            ), axis=1
        )
        
        return dados
    
    except Exception as e:
        print(f"Erro ao carregar os dados: {str(e)}")
        return None

def encontra_reacao(dados, nome, presente):
    presente_normalizado = normaliza_texto(presente)
    
    try:
        personagem = dados[dados['Name'] == nome].iloc[0]
    except IndexError:
        return None
    
    categorias = [
        ('Ama', 'Loved Gifts_normalizado'),
        ('Gosta', 'Liked Gifts_normalizado'),
        ('Neutro', 'Neutral Gifts_normalizado'),
        ('Não gosta', 'Disliked Gifts_normalizado'),
        ('Odeia', 'Hated Gifts_normalizado')
    ]
    
    for reacao, coluna in categorias:
        if presente_normalizado in personagem[coluna]:
            return reacao
    
    return None

def recomenda_presente(dados, presente):
    if dados is None or len(dados) == 0:
        return pd.DataFrame()
    
    todos_presentes = dados.apply(
        lambda linha: linha['Loved Gifts'] + linha['Liked Gifts'] + 
                     linha['Neutral Gifts'] + linha['Disliked Gifts'] + 
                     linha['Hated Gifts'], axis=1
    )
    
    dados['todos_presentes_str'] = todos_presentes.apply(
        lambda x: ' '.join([normaliza_texto(item) for item in x])
    )
    
    vetorizador = TfidfVectorizer(lowercase=False)
    matriz_tfidf = vetorizador.fit_transform(dados['todos_presentes_str'])
    
    presente_processado = normaliza_texto(presente)
    
    try:
        vetor_presente = vetorizador.transform([presente_processado])
    except ValueError:
        dados['similaridade'] = 0.0
    else:
        dados['similaridade'] = cosine_similarity(vetor_presente, matriz_tfidf)[0]
    
    dados['reacao'] = dados['Name'].apply(
        lambda nome: encontra_reacao(dados, nome, presente)
    )
    
    resultados = dados[dados['reacao'].notna()].copy()
    
    if not resultados.empty:
        ordem_reacao = {'Ama': 0, 'Gosta': 1, 'Neutro': 2, 'Não gosta': 3, 'Odeia': 4}
        resultados['ordem'] = resultados['reacao'].map(ordem_reacao)
        resultados = resultados.sort_values(['ordem', 'similaridade'], ascending=[True, False])
    
    return resultados

def mostra_resultados(resultados, presente):
    if resultados.empty:
        print(f"\nNão encontrado '{presente}'")
        return
    
    print(f"\nRESULTADOS PARA: {presente.upper()}")
    print("=" * 50)
    
    categorias = ['Ama', 'Gosta', 'Neutro', 'Não gosta', 'Odeia']
    
    for categoria in categorias:
        resultados_categoria = resultados[resultados['reacao'] == categoria]
        
        print(f"\n{categoria.upper()}")
        print("-" * len(categoria))
        
        if not resultados_categoria.empty:
            for _, personagem in resultados_categoria.iterrows():
                print(f"\n{personagem['Name']}")
                print(f"Aniversário: {personagem['Birthday Season']} dia {personagem['Birthday Day']}")
                print(f"Local: {personagem['Lives In']}")
                
                familia = ', '.join(personagem['Family']) if personagem['Family'] else 'Nenhuma'
                print(f"Família: {familia}")
                
                pode_casar = 'Sim' if personagem['Marriage'] == 'Yes' else 'Não'
                print(f"Pode casar: {pode_casar}")
                
                print(f"Afinidade: {personagem['similaridade']:.2f}")
        else:
            print("Nenhum personagem nesta categoria")

def main():
    print("\n" + "=" * 50)
    print("Sistema de Recomendação de Presentes do Stardew Valley".center(50))
    print("=" * 50)
    print("\nDigite o nome exato do presente como aparece no jogo em inglês")
    print("Exemplos: 'Daffodil', 'Tea Leaves', 'Pufferfish'")
    
    dados = carrega_dados()
    if dados is None:
        print("\nErro: Não foi possível carregar os dados dos personagens.")
        return
    
    while True:
        presente = input("\nDigite um presente: ").strip()
        
        if normaliza_texto(presente) == 'sair':
            break
            
        if not presente:
            print("Por favor, digite o nome de um presente.")
            continue
            
        resultados = recomenda_presente(dados, presente)
        mostra_resultados(resultados, presente)

if __name__ == "__main__":
    import sys
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPrograma interrompido pelo usuário.")
        sys.exit(0)
    except Exception as e:
        print(f"\nOcorreu um erro inesperado: {str(e)}")
        sys.exit(1)