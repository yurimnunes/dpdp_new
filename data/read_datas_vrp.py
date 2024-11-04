import pickle
import os


# Caminho da pasta que você quer iterar
folder_path_tsp = 'data/tsp'
folder_path_vrp = 'data/vrp'


##### Nao rodar
##### alguns arquivos podem consumir toda a ram
# o mais interessante seria transformar em csv
def peackle_to_txt(folder_path, output_path = None):
    
    if output_path is None:
        output_path = folder_path
    
    # Lista todos os arquivos na pasta
    files = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(output_path, filename)
        if os.path.isfile(file_path):
            files.append(file_path)

    for path in files:
        print(f"Processando arquivo: {path}")
        with open(path, 'rb') as f:
            data = pickle.load(f)
            # Processa o conteúdo do arquivo
        with open(path.replace('.pkl', '.txt'), 'w') as f:
            f.write(str(data))

#peackle_to_txt(folder_path_tsp)
#peackle_to_txt(folder_path_vrp)

def just_file_pickle_t_txt(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        print(data)
        f.close()
        # Processa o conteúdo do arquivo
    #with open(file_path.replace('.pkl', '.txt'), 'w') as f:
     #   f.write(str(data))
      #  f.close()
f1 = 'data/vrp/vrp_uchoa100_train_seed42.pkl'
f2 = 'data/vrp/vrp_uchoa100_validation_seed4321.pkl'


#### ALGUNS ARQUIVOS CONSOMEM TODA A RAM
#just_file_pickle_t_txt(f1)    
#just_file_pickle_t_txt(f2)
