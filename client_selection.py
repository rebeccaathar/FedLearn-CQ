
min_ber = 0.51855
max_no = 0.40

def client_selection(num_clients, metrics):
    print(metrics)
    clients_selected_indice = []
    for i in range(num_clients):
        print(f'client{i}: BER = {metrics(i)[0]}  SNR = {metrics(i)[1]} NO = {metrics(i)[2]}')
        if metrics(i)[0] <= min_ber and metrics(i)[2] <= max_no: #SNR and RSSI
            clients_selected_indice.append(i)
    
    print(f'Clients selected indices: {clients_selected_indice}')
    
    return clients_selected_indice
