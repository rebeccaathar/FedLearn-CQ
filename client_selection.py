from util import calculate_upload_time
from cdl_channel import cdl_channel_user
import tensorflow as tf
import numpy 

'''
ebno_db, no , data_rate, total_time = cdl_channel_user(int(cid)+1)

Total time = broadcast_time, local_training + upload_time

broadcast (perfect channel) = parameter_size/data_rate

local_training = 

upload_time = data_size(177704)/data_rate

'''
downlink_data_rate = 50e9 #50Gbit/s
server_param_data_size = 355408
client_param_data_size = 355408
local_training = 4.5630

min_snr = 9
min_total_time = 5.90

def client_selection(num_clients, metrics):
    clients_selected_indice = []
    for i in range(num_clients):
        snr = metrics(i)[0]
        time_broadcast = server_param_data_size/downlink_data_rate
        uplink_data_rate = metrics(i)[2]
        upload_time = client_param_data_size/uplink_data_rate
        total_time = time_broadcast + local_training +  upload_time
        noise_level = metrics(i)[1]

        #print(f'client{i}: snr = {snr}  uplink_data_rate = {uplink_data_rate / 1e6:.3f} Mbps  upload_time = {upload_time:.3f}s  total_time = {total_time:.3f}s ')

        if snr >= min_snr and total_time <= min_total_time: 
            clients_selected_indice.append(i)
    
    print(f'Clients selected indices: {clients_selected_indice}')
    
    return clients_selected_indice


def clients_metrics(num_clients, metrics):
    clients_selected_info = []
    for i in range(num_clients):
        snr = metrics(i)[0]
        time_broadcast = server_param_data_size/downlink_data_rate
        uplink_data_rate = metrics(i)[2]
        upload_time = client_param_data_size/uplink_data_rate
        total_time = time_broadcast + local_training + upload_time
        noise_level = metrics(i)[1].numpy().item()
        
        clients_selected_info.append({
                "id": i,
                "snr": snr,
                "uplink_data_rate": uplink_data_rate/ 1e6,
                "upload_time": upload_time,
                "total_time": total_time,
                "noise_level": noise_level
            })
    
    return clients_selected_info



def filter_clients(client_metrics, min_snr=8, max_time=5):
    # Inicializa uma lista vazia para armazenar os índices dos clientes filtrados
    indices_filtrados = []
    
    # Itera sobre os dados de cada cliente
    for i, cliente in enumerate(client_metrics):
        if cliente['snr'] >= min_snr and cliente['total_time'] <= max_time:
            # Adiciona o índice do cliente à lista de índices filtrados
            indices_filtrados.append(str(i))
            # print(indices_filtrados)
    
    return indices_filtrados
