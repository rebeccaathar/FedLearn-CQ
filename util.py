import numpy as np
import sys
import yaml

def get_data_size(data_list):
    total_size = 0
    for sublist in data_list:
        # Se os elementos são listas ou tuplas
        if isinstance(sublist, (list, tuple)):
            # Recursivamente calcula o tamanho da sublista
            total_size += get_data_size(sublist)
            # Se os elementos são arrays numpy ou outras estruturas de dados
            # que têm um método `nbytes` para calcular o tamanho em bytes
        elif hasattr(sublist, 'nbytes'):
            total_size += sublist.nbytes
            # Se os elementos são outros tipos de dados (como números ou strings)
        else:
            total_size += sys.getsizeof(sublist)
            
    return total_size



def delay_broadcast(self, parameters_received):
               
    a = self.set_parameters(parameters_received)

    #========= delay broadcast ===========
    # Tamanho de um float32 em bytes
    float32_size = 4

    # Calculando o número total de float32 na lista
    total_floats = sum(len(sublist) for sublist in parameters_received)

    # Calculando o tamanho total em bytes
    total_size_in_bytes = total_floats * float32_size

    # delay_broadcast = total_size_in_bytes/data_rate
    return delay_broadcast


# def add_noise(params, noise_level):
#         '''adding noise to the parameters'''
#         return [param + np.random.normal(0, noise_level, size=param.shape) for param in params]


def calculate_data_rate(bandwidth, snr_db, num_bits_per_symbol, coderate):
    """
    Calcula a capacidade do canal usando a fórmula de Shannon-Hartley.

    Parâmetros:
    bandwidth (Hz): Largura de banda do canal em Hz
    snr_db (dB): Relação Sinal-Ruído em decibéis
    spectral_efficiency (bit/s/Hz): Eficiência espectral

    Retorna:
    capacidade (bps): Capacidade do canal em bits por segundo
    """

    spectral_efficiency = num_bits_per_symbol * coderate

    # Converter SNR de dB para valor linear
    snr_linear = 10 ** (snr_db / 10)
    
    # Calcular a capacidade do canal usando a fórmula de Shannon-Hartley
    capacity = bandwidth * np.log2(1 + snr_linear)
    
    # Ajustar a capacidade pela eficiência espectral
    data_rate = capacity * spectral_efficiency
    
    return data_rate

def calculate_upload_time(data_size, data_rate):
    return data_size/data_rate


def add_noise(data_list, noise_level):
    """
    Adiciona ruído a uma lista de arrays NumPy.

    Parameters:
    data_list (list): Lista contendo arrays NumPy.
    noise_level (float): Nível de ruído a ser adicionado.

    Returns:
    list: Nova lista com os arrays ruidosos.
    """
    noisy_data_list = []
    for array in data_list:
        noise = np.random.normal(0, noise_level, array.shape)
        noisy_array = array + noise
        noisy_data_list.append(noisy_array)
    return noisy_data_list


def update_base(clients_selected_indice):
    file_path = 'conf/base.yaml'

    # Carrega a configuração do arquivo YAML
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)

    # Atualiza o valor de clients_selected_indice no dicionário de configuração
    config['clients_selected_indice'] = [str(client) for client in clients_selected_indice]

    # Salva o dicionário de volta no arquivo YAML
    with open(file_path, 'w') as file:
        yaml.safe_dump(config, file)
    