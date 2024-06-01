

min_ber = 0.1855

def client_selection(num_clients, metrics):
    print(metrics)
    clients_selected_indice = []
    for i in range(num_clients):
        print(f'client{i+1}: BER = {metrics(i)[0]}  SNR = {metrics(i)[1]} NO = {metrics(i)[2]}')
        if metrics(i)[0] <= min_ber: 
            clients_selected_indice.append(i)
    
    print(f'Clients selected indices: {clients_selected_indice}')
    
    return clients_selected_indice
