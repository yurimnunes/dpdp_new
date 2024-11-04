# -*- coding: utf-8 -*-

import vrplib

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

def le_cvrptw(instancia:str, solucao:str):    
            
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
import sys
instancia = ("./data/Vrp-Set-XML100/instances/XML100_1111_01.vrp",
             "./data/Vrp-Set-XML100/solutions/XML100_1111_01.sol")
instancia, solucao = le_instancia_solucao(instancia[0], instancia[1])
                                          
x = le_cvrptw(instancia, solucao)

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

rotas_da_solucao = solucao['routes']
custo_da_solucao = solucao['cost']

# 1) Geração de features dos nós:
# 1.1) Feature coordenadas geográficas (distância Euclidiana)
maior = 0
for i in range(dimensao_instancia):
    if nos_coordenadas[i][0] > maior:
        maior = nos_coordenadas[i][0]
    if nos_coordenadas[i][1] > maior:
        maior = nos_coordenadas[i][1]
 # normalização das coordenadas x e y
feature_coor = []
for i in range(dimensao_instancia):
    no = []
    x = nos_coordenadas[i][0] / maior
    y = nos_coordenadas[i][1] / maior
    no.append(x)
    no.append(y)
    feature_coor.append(no)

# 1.2) Feature carga a ser retirada no nó
# normalizando pela capacidade do veículo
feature_capac = []
for i in range(dimensao_instancia):
    cap = nos_demanda[i] / capacidade_veiculo
    feature_capac.append(cap)

# 2) Geração do mapa geral de distância entre todos os nós.
mapa_coord = []
for i in range(dimensao_instancia):
    linha = []
    for j in range(dimensao_instancia):
        dist_arco = peso_arcos[i][j] / maior
        linha.append(dist_arco)
    mapa_coord.append(linha)

# 3) Valor ótimo do custo:
custo = custo_da_solucao / maior

# 4) Dados de saída:
heatmap = []
for i in range(dimensao_instancia):
    linha = []
    for j in range(dimensao_instancia):
        linha.append(0)
    heatmap.append(linha)
cont = 0
for i in range(len(rotas_da_solucao)):
    anterior = 0
    for j in range(len(rotas_da_solucao[i])):
        heatmap[anterior][rotas_da_solucao[i][j]] = 1
        anterior = rotas_da_solucao[i][j]
    heatmap[anterior][0] = 1

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

