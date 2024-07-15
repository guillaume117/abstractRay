import torch
import torch.nn as nn
import sys
sys.path.append('app/src')
sys.path.append('./src')
from zono_sparse_gen_2 import ZonoSparseGeneration
from util import sparse_tensor_stats, get_largest_tensor_size,sparse_dense_broadcast_mult, resize_sparse_coo_tensor
from process_layer import process_layer



class ModelEvaluator:

    def __init__(self, unstacked_model, abstract_domain, num_workers = 0,available_RAM = 8, device =torch.device('cpu'),add_symbol =True,renew_abstract_domain=False,verbose = False):
        
        self.model_unstacked = unstacked_model
        self.abstract_domain = abstract_domain
        self.num_workers = num_workers
        self.available_RAM = available_RAM
        self.device = device
        self.add_symbol = add_symbol
        self.verbose = verbose
        self.renew_abstract_domain =renew_abstract_domain

    def evaluate_model(self):

        for name, details in self.model_unstacked.items():
            self.name = name
            self.details = details
            if self.verbose:
                print('*'*100)
                print(f'before layer {self.name}')
                print(self.abstract_domain['center'].size())
                print(self.abstract_domain['zonotope'].size())
                print(self.abstract_domain['sum'].size())
                print(self.abstract_domain['trash'].size())
                print(self.abstract_domain['perfect_domain'])


            self.abstract_domain = process_layer(self.abstract_domain,self.name,self.details,self.num_workers, self.available_RAM,self.device,self.add_symbol)
            if self.verbose:
                print('*'*100)
                print(f'after layer {self.name}')
                print(self.abstract_domain['center'].size())
                print(self.abstract_domain['zonotope'].size())
                print(self.abstract_domain['sum'].size())
                print(self.abstract_domain['trash'].size())
                print(self.abstract_domain['perfect_domain'])
            if self.renew_abstract_domain and self.abstract_domain['perfect_domain']==True:
                new_symbs= 2*self.abstract_domain['sum'].to_dense()

                new_symbs_sparse = new_symbs.to_sparse_coo().coalesce()
                print(new_symbs_sparse)
                self.abstract_domain['zonotope'] = ZonoSparseGeneration().zono_from_tensor(new_symbs_sparse).coalesce()
                print(self.abstract_domain['zonotope'])
                self.abstract_domain['trash'] = torch.zeros_like(new_symbs)

        return self.abstract_domain           