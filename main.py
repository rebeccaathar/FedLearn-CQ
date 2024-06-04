import pickle
from pathlib import Path
import numpy as np 
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import flwr as fl

from dataset import prepare_dataset
from client import generate_client_fn
from client_selection import client_selection
from server import get_on_fit_config, get_evaluate_fn
from util import update_base

# A decorator for Hydra. This tells hydra to by default load the config in conf/base.yaml
@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    # np.random.seed(42)
    '''1. Parse config & get experiment output dir'''
    print(OmegaConf.to_yaml(cfg))
    
    save_path = HydraConfig.get().runtime.output_dir

    '''2. Prepare your dataset'''
    trainloaders, validationloaders, testloader = prepare_dataset(
        cfg.num_clients, cfg.batch_size
    )

    # print(len(trainloaders), len(trainloaders[0]).dataset)
    # 100 clients, so 100 dataloaders, size of the dataset of the first dataloader

    '''3. Define your clients and channel metrics'''
    #(aqui é como se distribuísse os modelos entre os clientes??)
    #Return a function that will be able to instant data client of a particular account ID
    #Thats the way the virtual client engine will create his clients at the beginning of every round if the client ID with
    # a particular ID is told to participate. 

    #A function creating client instances. The function must take a single str argument called cid. 
    #It should return a single client instance of type Client.

    client_fn , metrics = generate_client_fn(trainloaders, validationloaders, cfg.num_classes)

    ''' 2. Filtrar os clientes '''    

    clients_selected_indice = client_selection(cfg.num_clients, metrics)
    
    ''' 3. Atualizar o arquivo base.yaml '''
    update_base(clients_selected_indice)

    '''4. Define your strategy'''
    
    '''Strategy Using FedAvg'''
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,  # in simulation, since all clients are available at all times, we can just use `min_fit_clients` to control exactly how many clients we want to involve during fit
        #min_fit_clients=1,  # number of clients to sample for fit()
        fraction_evaluate=1.0,  # similar to fraction_fit, we don't need to use this argument.
        #min_evaluate_clients=cfg.num_clients_per_round_eval,  # number of clients to sample for evaluate()
        accept_failures = False,
        # min_available_clients = 11,
        min_available_clients= 2,
        #min_available_clients=cfg.num_clients,  # total clients available
        on_fit_config_fn=get_on_fit_config(
            cfg.config_fit
        ),  # a function to execute to obtain the configuration to send to the clients during fit()
        evaluate_fn=get_evaluate_fn(cfg.num_classes, testloader),
    )  # a function to run on the server side to evaluate the global model.

    '''5. Start Simulation'''
    '''#=dá para setar o round timeout!!'''
    # With the dataset partitioned, the client function and the strategy ready, we can now launch the simulation!
    history = fl.simulation.start_simulation(
        client_fn=client_fn,  # a function that spawns a particular client - função que cria os clientes
        #num_clients=cfg.num_clients,  # total number of clients available --> len(filter_client)
        #num_clients = len(clients_selected_indice)
        clients_ids = cfg.clients_selected_indice,
        config=fl.server.ServerConfig(
                    num_rounds=cfg.num_rounds),  # minimal config for the server loop telling the number of rounds in FL
        strategy=strategy,  # our strategy of choice
        client_resources={
            "num_cpus": 4,
            "num_gpus": 0.0,
        }, 
    )

    '''6. Save your results'''
    results_path = Path(save_path) / "results.pkl"

    results = {"history": history, "anythingelse": "here"}

    # save the results as a python pickle
    with open(str(results_path), "wb") as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
