# -*- coding: utf-8 -*-

import vrplib
import numpy as np


def le_instancia_solucao(instancia, solucao):
    
    #Origem da informação sobre benchmarking: 
    #    http://vrp.galgos.inf.puc-rio.br/index.php/en/
    #    pasta: XML100 (2021)
    #Os arquivos foram colocados na pasta (local) "Vrp-Set-XML100"
    #Uso do vrplib: https://pypi.org/project/vrplib/
    
    
    # Leitura de instâncias no padrão Solomon:
    instance = vrplib.read_instance(instancia)
    solution = vrplib.read_solution(solucao)
    
    #instance and solution are dictionaries that contain all parsed data.
    #>>> instance.keys()
    #dict_keys(['name', ..., 'edge_weight'])
    
    #>>> solution.keys()
    #dict_keys(['routes', 'cost'])
    
    return (instance, solution)
#----------------------------------------------------------------------------
#Origem da documentação sobre CVRP/CRVPTW: https://arxiv.org/pdf/1606.01935
#Programa para tratamento das variáveis: https://github.com/sudhan-bhattarai/
#   CVRPTW_MILP/blob/master/CVRPTW.py

def le_cvrptw(instancia, solucao):    
            
    nome_da_instancia = instancia['name']
    comentario_instancia = instancia['comment']
    tipo_instancia = instancia['type']
    dimensao_instancia = instancia['dimension']
    tipo_custo = instancia['edge_weight_type']
    capacidade_veiculo = instancia['capacity']
    nos_coordenadas = instancia['node_coord']
    nos_demanda = instancia['demand']
    deposito_instancia = instancia['depot']
    peso_arcos = instancia["edge_weight"]

    
    print("NAME: ", nome_da_instancia)
    print("COMMENT: ", comentario_instancia)
    print("TYPE: ", tipo_instancia)
    print("DIMENSION: ", dimensao_instancia)
    print("EDGE_WEIGHT_TYPE: ", tipo_custo)
    print("CAPACITY: ", capacidade_veiculo)
    print("NODE_COORD_SECTION")
    for i in range(1, dimensao_instancia + 1):
        print(i, nos_coordenadas[i - 1][0], nos_coordenadas[i - 1][1])
    print("DEMAND_SECTION")
    for i in range(1, dimensao_instancia + 1):
        print(i, nos_demanda[i -1])
    print("DEPOT_SECTION")
    print(deposito_instancia[0])
    print() ##
    print("EDGE_WEIGHT_SECTION")
    for i in range(1, len(peso_arcos) + 1):
        print(i, " ", end = "")
        for j in range(4):
            print(peso_arcos[i - 1][j], " ", end = "" )
        print()
    print("EOF")
    print()

    rotas_da_solucao = solucao['routes']
    custo_da_solucao = solucao['cost']
    for i in range(len(rotas_da_solucao)):
        rota = "Route #" + str(i + 1) + ": "
        print(rota, end = '')
        for j in range(len(rotas_da_solucao[i])):
            print(rotas_da_solucao[i][j], ' ', end = '')
        print()
    print("Cost", custo_da_solucao)
    
    
#------------------------------------------------------------------------------
import os
dados_entrada_gnn = []
diretorio_inst = "./data/Vrp-Set-XML100/instances/"
lista_arquivos = os.listdir(diretorio_inst)

#############################
for ii in range(len(lista_arquivos)):
#for ii in range(1000):
    #print(ii)
    inst =  diretorio_inst + lista_arquivos[ii]
    instancia = vrplib.read_instance(inst)

    # A partir deste ponto serão gerados todos os dados necessários à Rede Neural.
    nome_da_instancia = instancia['name']
    comentario_instancia = instancia['comment']
    tipo_instancia = instancia['type']
    dimensao_instancia = instancia['dimension']
    tipo_custo = instancia['edge_weight_type']
    capacidade_veiculo = instancia['capacity']
    nos_coordenadas = instancia['node_coord']
    nos_demanda = instancia['demand']
    deposito_instancia = instancia['depot']
    peso_arcos = instancia["edge_weight"]
    
    #rotas_da_solucao = solucao['routes']
    #custo_da_solucao = solucao['cost']
    
    # 0) Cria tupla para instancia corrente:
    tupla_entrada = ([], [], [], capacidade_veiculo)
    #tupla_entrada = [[], [], [], capacidade_veiculo]
    
    # 1) Geração de features dos nós:
    # 1.1) Feature coordenadas geográficas (distância Euclidiana)
    # do depot e de todos os nós:
    grid_size = 1000
    # normalização das coordenadas x e y
    x_normalizado = nos_coordenadas[0][0] / grid_size
    y_normalizado = nos_coordenadas[0][1] / grid_size
    tupla_entrada[0].append(x_normalizado)
    tupla_entrada[0].append(y_normalizado)
    
    # normalização das coordenadas x e y
    for i in range(1, dimensao_instancia):
        no = []
        tupla_entrada[1].append(no)
        x_normalizado = nos_coordenadas[i][0] / grid_size
        y_normalizado = nos_coordenadas[i][1] / grid_size
        tupla_entrada[1][i - 1].append(x_normalizado)
        tupla_entrada[1][i - 1].append(y_normalizado)
    # 1.2) Feature carga a ser retirada no nó
    # este item não precisa de normalização
    for i in range(1, dimensao_instancia):
        cap = nos_demanda[i]
        tupla_entrada[2].append(cap)
    
    # 2) Geração da informação da capacidade do caminhão. Esta
    # informação não é normalizada. Foi gerada na criação da tupla.
    # capacidade_veiculo = instancia['capacity']
    
    dados_entrada_gnn.append(tupla_entrada)
    del(tupla_entrada)


entrada_gnn_numpy = np.array(dados_entrada_gnn, dtype=object)
# Imprime dados de instâncias
"""
for i in range(len(lista_arquivos)):
    #print(dados_entrada_gnn[i])
    print(entrada_gnn_numpy[i])
    #print(lista_arquivos[i])
    print()
"""
# Suponha que array_dados já contenha seus dados
n_total = len(entrada_gnn_numpy)

# Índices para dividir os dados
train_end = int(0.7 * n_total)     # 70%
val_end = int(0.9 * n_total)       # 90%

# Dividir os dados
dados_treino = entrada_gnn_numpy[:train_end]
#dados_validacao = entrada_gnn_numpy[train_end:val_end]
dados_validacao = entrada_gnn_numpy[train_end:train_end+10]
dados_teste = entrada_gnn_numpy[val_end:]

# Salvar cada parte em arquivos .npy
np.save('dados_treino.npy', dados_treino)
np.save('dados_validacao.npy', dados_validacao)
np.save('dados_teste.npy', dados_teste)

print("Dados salvos em arquivos .npy com sucesso!")


# A partir daqui vai gerar o arquivo com todas as soluções possíveis:
dados_saida_gnn = []
diretorio_solutions = "./data/Vrp-Set-XML100/solutions/"
lista_arquivos = os.listdir(diretorio_solutions)
for ii in range(len(lista_arquivos)):
#for ii in range(1000):
    sol = diretorio_solutions + lista_arquivos[ii]
    solucao = vrplib.read_solution(sol)

    # A partir deste ponto serão gerados todos os dados necessários à Rede Neural.
   
    rotas_da_solucao = solucao['routes']
    #custo_da_solucao = solucao['cost']
    
    # 0) Cria lista para instancia corrente:
    lista_saida = []
    
    # 1) Geração das rotas dos nós:
    for i in range(len(rotas_da_solucao)):
        rotas_da_solucao = solucao['routes']
        for j in range(len(rotas_da_solucao[i])):
            lista_saida.append(rotas_da_solucao[i][j])
        if i < len(rotas_da_solucao) - 1:
            lista_saida.append(0)
    dados_saida_gnn.append(lista_saida)
    del(lista_saida)


dados_saida_gnn_numpy = np.array(dados_saida_gnn, dtype=object)
    
"""
for i in range(len(lista_arquivos)):
    print(dados_saida_gnn[i])
    print(lista_arquivos[i])
    print()
"""

# Suponha que array_dados já contenha seus dados
n_total = len(dados_saida_gnn_numpy)

# Índices para dividir os dados
train_end = int(0.7 * n_total)     # 70%
val_end = int(0.9 * n_total)       # 90%

# Dividir os dados
dados_solucao_treino = dados_saida_gnn_numpy[:train_end]
#dados_solucao_validacao = dados_saida_gnn_numpy[train_end:val_end]
dados_solucao_validacao = dados_saida_gnn_numpy[train_end:train_end+10]
dados_solucao_teste = dados_saida_gnn_numpy[val_end:]

# Salvar cada parte em arquivos .npy
np.save('dados_solucao_treino.npy', dados_solucao_treino)
np.save('dados_solucao_validacao.npy', dados_solucao_validacao)
np.save('dados_solucao_teste.npy', dados_solucao_teste)

print("Dados salvos em arquivos .npy com sucesso!")







# Imprime heatmap:
'''
for i in range(dimensao_instancia):
    print(i, " :", end = "")
    for j in range(dimensao_instancia):
        if heatmap[i][j] == 1:
            print(j, " ", end = "")
    print()    
'''      

'''
#------------------------------------------------------------------------------   
# Manipula todas as 10.000 instâncias:
def le_rotas_solucao(solucao):
    solution = vrplib.read_solution(solucao)
    quantidade_rotas_da_solucao = len(solution['routes'])
    return quantidade_rotas_da_solucao
#------------------------------------------------------------------------------   
def le_capacity_instancia(instancia):
    instance = vrplib.read_instance(instancia)
    capacity_inst = instance['capacity']
    return capacity_inst
#------------------------------------------------------------------------------   
# Teste: quantidade de instâncias por quantidade de rotas.
import os
diretorio = "./Vrp-Set-XML100/solutions/"
lista_arquivos = os.listdir(diretorio)
print(len(lista_arquivos))

#Teste: quantidade de instâncias por capacity
import os
diretorio = "./Vrp-Set-XML100/instances/"
lista_arquivos = os.listdir(diretorio)
print(len(lista_arquivos))
lista_quantidade_cap = []
for i in range (13001):
    lista_quantidade_cap.append(0)
for i in range(len(lista_arquivos)):
    inst = "./Vrp-Set-XML100/instances/" + lista_arquivos[i]
    total_por_cap = le_capacity_instancia(inst)
    lista_quantidade_cap[total_por_cap] += 1
soma = 0
for i in range(13001):
    if lista_quantidade_cap[i] != 0:
        soma += lista_quantidade_cap[i]
        perc = lista_quantidade_cap[i] / 100
        #print("Arquivos com capacidade", i, ": ", lista_quantidade_cap[i], "...", perc, "%")
print()
print(soma)
'''

