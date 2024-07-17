import torch
import torch.nn as nn
import sys
import time
import json

sys.path.append('app/src')
sys.path.append('./src')
from zono_sparse_gen_2 import ZonoSparseGeneration
from util import sparse_tensor_stats, get_largest_tensor_size, sparse_dense_broadcast_mult, resize_sparse_coo_tensor
from process_layer import process_layer

class ModelEvaluator:

    def __init__(self, unstacked_model, abstract_domain, num_workers=0, available_RAM=8, device=torch.device('cpu'), add_symbol=True, renew_abstract_domain=False, verbose=False, json_file_prefix='evaluation_results',noise_level=0.00001):
        self.model_unstacked = unstacked_model
        self.abstract_domain = abstract_domain
        self.num_workers = num_workers
        self.available_RAM = available_RAM
        self.device = device
        self.add_symbol = add_symbol
        self.verbose = verbose
        self.renew_abstract_domain = renew_abstract_domain
        self.json_file_prefix = json_file_prefix
        self.timestart = time.time()
        self.noise_level = noise_level
        self.num_symbols = abstract_domain['zonotope']._nnz()
        self.input_tensor_size_init = tuple(self.abstract_domain['center'].size())

    def evaluate_model(self):
        all_results = {
            "context": {
                "model_name": self.json_file_prefix,
                "num_workers": self.num_workers,
                "available_RAM": self.available_RAM,
                "device": str(self.device),
                "add_symbol": self.add_symbol,
                "renew_abstract_domain": self.renew_abstract_domain,
                "verbose": self.verbose,
                "noise_level": self.noise_level,
                "num_symbols": self.num_symbols,
                "input_tensor_size": self.input_tensor_size_init,
                "process_ended":False,
            },
            "layers": []
        }

        for name, details in self.model_unstacked.items():
            self.name = name
            self.details = details
            start_time = time.time()

            if self.verbose:
                print('*' * 100)
                print(f'before layer {self.name}')
                print(self.abstract_domain['center'].size())
                print(self.abstract_domain['zonotope'].size())
                print(self.abstract_domain['sum'].size())
                print(self.abstract_domain['trash'].size())
                print(self.abstract_domain['perfect_domain'])

            self.abstract_domain = process_layer(self.abstract_domain, self.name, self.details, self.num_workers, self.available_RAM, self.device, self.add_symbol)

            if self.verbose:
                print('*' * 100)
                print(f'after layer {self.name}')
                print(self.abstract_domain['center'].size())
                print(self.abstract_domain['zonotope'].size())
                print(self.abstract_domain['sum'].size())
                print(self.abstract_domain['trash'].size())
                print(self.abstract_domain['perfect_domain'])

            num_symbols = self.abstract_domain['zonotope'].size(0)
            nnz = self.abstract_domain['zonotope']._nnz()
            non_zero_percentage, memory_gain_percentage = sparse_tensor_stats(self.abstract_domain['zonotope'])
            end_time = time.time()
            computation_time = end_time - start_time

            # Convertir les informations de taille du tenseur en tuple
            input_tensor_size = tuple(self.abstract_domain['center'].size())

            layer_evaluation = {
                "layer_name": self.name,
                "layer_details" : self.details,
                "input_tensor_size": input_tensor_size,
                "num_symbols": num_symbols,
                "nnz": nnz,
                "memory_gain_percentage": float(memory_gain_percentage),
                "computation_time": computation_time,
                #"abstract_domain_center" : self.abstract_domain['center'].tolist(),
                #"max_bounds":self.abstract_domain['sum'].tolist()
            }

            all_results["layers"].append(layer_evaluation)

            

            if self.renew_abstract_domain and self.abstract_domain['perfect_domain']:
                new_symbs = 2 * self.abstract_domain['sum'].to_dense()
                new_symbs_sparse = new_symbs.to_sparse_coo().coalesce()
                self.abstract_domain['zonotope'] = ZonoSparseGeneration().zono_from_tensor(new_symbs_sparse).coalesce()
                self.abstract_domain['trash'] = torch.zeros_like(new_symbs)
            self._save_results_to_json(all_results)
        all_results['context']['process_ended']=True
        self._save_results_to_json(all_results)
        return self.abstract_domain

    def _save_results_to_json(self, all_results):


        json_file = f'../result/{self.json_file_prefix}_time_eval{self.timestart}.json'
        with open(json_file, 'w') as f:
            json.dump(all_results, f, indent=4, default=str)  # Utilisation de default=str pour sérialiser des objets non-standard

# Utilisation de la classe ModelEvaluator
# unstacked_model et abstract_domain doivent être définis avant cette utilisation
# evaluator = ModelEvaluator(unstacked_model, abstract_domain, verbose=True)
# abstract_domain = evaluator.evaluate_model()
