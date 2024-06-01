from collections import OrderedDict
from typing import Dict, Tuple
from flwr.common import NDArrays, Scalar
from cdl_channel import cdl_channel_user
import numpy as np
import random
import torch
import flwr as fl
import time
import sys
from cdl_channel import cdl_channel_user
import time
from model import Net, train, test

#Cliente has two main methods: fit() and evaluate()
#fit: receive the current Global model and train it using the local dataset ans then it sends it back to the server where it will be aggregated. 
#Evaluate: evaluating the state of the global model on the local data distribution of the client

#in order to run both fit and evaluate weneed two axillary methods:
# set_parameters: that have copies the parameter sent by the server into your model representation
# get_parameters: thats just the opposite it extracts the weight from your model and represents a list of numpy array

'''
def to_client():
    fi


'''


class FlowerClient(fl.client.NumPyClient):
    """Define a Flower Client."""
    
    client_counter = 0

    def __init__(self, trainloader, vallodaer, num_classes) -> None:
        super().__init__()

        # the dataloaders that point to the data associated to this client
        self.trainloader = trainloader
        self.valloader = vallodaer

        # For further flexibility, we don't hardcode the type of model we use in
        # federation. Here we are instantiating the object defined in `conf/model/net.yaml`
        # (unless you changed the default) and by then `num_classes` would already be auto-resolved
        # to `num_classes=10` (since this was known right from the moment you launched the experiment)

        # a model that is randomly initialised at first
        self.model = Net(num_classes)

        # figure out if this client has access to GPU support or not
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def add_noise(self, params, noise_level):
        '''adding noise to the parameters'''
        return [param + np.random.normal(0, noise_level, size=param.shape) for param in params]


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
        # return delay_broadcast


    #downlink
    def set_parameters(self, parameters):#parameters that we get from the server
        """ Receive parameters and apply them to the local model."""
        
        #Converting every element in a numpy array into a pytorch tensor presentation
        params_dict = zip(self.model.state_dict().keys(), parameters)

        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})

        self.model.load_state_dict(state_dict, strict=True)

    # This method basically praped the process (set_parameters)
    # Earlier we have a list of numpy arrays that is what the server sent us that we copy those into the model
    # Now we need to do the opposite. Lets extract a list of number arrys from our model
    def get_parameters(self, config: Dict[str, Scalar]):
        """Extract model parameters and return them as a list of numpy arrays."""

        # first ensure that my variable is in the CPU and then we're gonna convert it to numpy.
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    #Train this model localy
    def fit(self, parameters, config):
        """
        Train model received by the server (parameters) using the data.

        that belongs to this client. Then, send it back to the server.
        """

        # if FlowerClient.client_counter <= 3:
        #     client = FlowerClient.client_counter
        # else:
        #     client = FlowerClient.client_counter - 3
        
        # FlowerClient.client_counter += 1

        FlowerClient.client_counter += 1

        # Garante que o número do cliente seja 1, 2 ou 3
        client = (FlowerClient.client_counter - 1) % 3 + 1
        
        #tf.seed() dá para usar só configurar que cada cliente tenha um id, tipo clien+=1

        # copy parameters sent by the server into client's local model
        self.set_parameters(parameters)
        # fetch elements in the config sent by the server. Note that having a config
        # sent by the server each time a client needs to participate is a simple but
        # powerful mechanism to adjust these hyperparameters during the FL process. For
        # example, maybe you want clients to reduce their LR after a number of FL rounds.
        # or you want clients to do more local epochs at later stages in the simulation
        # you can control these by customising what you pass to `on_fit_config_fn` when
        # defining your strategy.
        lr = config["lr"]
        momentum = config["momentum"]
        epochs = config["local_epochs"]

        # You could also set this optimiser from a config file. That would make it
        # easy to run experiments considering different optimisers and set one or another
        # directly from the command line (you can use as inspiration what we did for adding
        # support for FedAvg and FedAdam strategies)
        
        # a very standard looking optimiser
        optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

        '''do local training '''
        # This function is identical to what you might
        # have used before in non-FL projects. For more advance FL implementation
        # you might want to tweak it but overall, from a client perspective the "local
        # training" can be seen as a form of "centralised training" given a pre-trained
        # model (i.e. the model received from the server)
        # Record the start time
        start_time = time.time()
        train(self.model, self.trainloader, optim, epochs, self.device)
        # Record the end time
        end_time = time.time()
        
        # Calculate the training duration
        training_duration = end_time - start_time

        # Send back the updated model back to the server
        # Return a bit of information of how this dataset is, how many training examples this client used. 
        # We're going to be using a aggregation method called ferabrese and a version of it requires knowing how many 
        # training example wereused by every client. 

        ber, ebno_db, no = cdl_channel_user(client)
        print(f'Client: {client}, no: {no}')
     

        parameters_noisy = self.add_noise(self.get_parameters({}), no)

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

        ##O data size não variou mesmo com grandes alterações no noise level, permanecendo sempre com 355408 bytes
        print(f'SIZE DA LISTA: {get_data_size(parameters_noisy)}')
        # return self.get_parameters({}), len(self.trainloader), {}
        return parameters_noisy, len(self.trainloader), {}

        #return self.get_parameters({}), len(self.trainloader), {}

   

    # Received the global model from the server an the idea here is that we dont want to modify the global modeal we just want to evaluate how the global model performs on the validation.
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        #first copying the parameters into the local model 
        self.set_parameters(parameters)
        #we need the loss and the accuracy and we are going to report back to the server in order to have a sense of how the global model is doing 
        loss, accuracy = test(self.model, self.valloader, self.device)
        #Ensure that the loss is a float and send to server how many examples were in the validation roller

        return float(loss), len(self.valloader), {"accuracyyyy": accuracy}

#we need to pass a function that can be called by server in order to spawn a client
#spawn these clients 0r wake up these clients for simulation
#this function will pass a single argument which is the client ID that is supposed to participate in a given round 

def generate_client_fn(trainloaders, valloaders, num_classes):
    """Return a function that can be used by the VirtualClientEngine.

    to spawn a FlowerClient with client id `cid`.
    """

    def metrics(cid:str):
        channel_metrics = []
        ber, ebno_db, no = cdl_channel_user(int(cid)+1)
        channel_metrics.append(ber)
        channel_metrics.append(ebno_db)
        channel_metrics.append(no)

        return channel_metrics

    def client_fn(cid: str):
        # This function will be called internally by the VirtualClientEngine
        # Each time the cid-th client is told to participate in the FL
        # simulation (whether it is for doing fit() or evaluate())

        # Returns a normal FLowerClient that will use the cid-th train/val
        # dataloaders as it's local data.
      
        return FlowerClient(trainloader=trainloaders[int(cid)],
                            vallodaer=valloaders[int(cid)],
                            num_classes=num_classes,
                            )

    # return the function to spawn client
    return client_fn , metrics