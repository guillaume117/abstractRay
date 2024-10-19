import torch
import torch.nn as nn
import time
from tabulate import tabulate
from AbstractRay.backend.src.zono_sparse_gen import ZonoSparseGeneration
from AbstractRay.backend.src.util import sparse_tensor_stats
from AbstractRay.backend.src.process_layer import process_layer
from AbstractRay.result.save_results_to_json import save_results_to_json
from AbstractRay.backend.src.sparse_evaluation import SparseEvaluationParallel

class ModelEvaluator:
    """
    A class to evaluate a model with unstacked layers using a given abstract domain.

    Attributes:
        unstacked_model (dict): A dictionary containing the unstacked model layers and details.
        abstract_domain (dict): The abstract domain to be used for evaluation.
        num_workers (int): Number of workers for processing.
        available_RAM (int): Available RAM in GB.
        device (torch.device): The device to run the evaluation on (CPU or GPU).
        add_symbol (bool): Flag to indicate whether to add a symbol.
        renew_abstract_domain (bool): Flag to indicate whether to renew the abstract domain.
        verbose (bool): Flag to enable verbose output.
        json_file_prefix (str): Prefix for the JSON file to save results.
        noise_level (float): Noise level for the evaluation.
        num_symbols (int): Number of non-zero elements in the zonotope.
        input_tensor_size_init (tuple): Initial size of the input tensor.
        timestart (float): Start time of the evaluation process.
    """

    def __init__(self, unstacked_model, abstract_domain, num_workers=0, available_RAM=8, device=torch.device('cpu'), add_symbol=True, renew_abstract_domain=False, verbose=True, json_file_prefix='evaluation_results', noise_level=0.00001,model_cut =False,parrallel_rel=False):
        """
        Initialize the ModelEvaluator class.

        Args:
            unstacked_model (dict): The unstacked model to evaluate.
            abstract_domain (dict): The abstract domain to be used for evaluation.
            num_workers (int, optional): Number of workers for processing. Defaults to 0.
            available_RAM (int, optional): Available RAM in GB. Defaults to 8.
            device (torch.device, optional): The device to run the evaluation on. Defaults to torch.device('cpu').
            add_symbol (bool, optional): Flag to indicate whether to add a symbol. Defaults to True.
            renew_abstract_domain (bool, optional): Flag to indicate whether to renew the abstract domain. Defaults to False.
            verbose (bool, optional): Flag to enable verbose output. Defaults to False.
            json_file_prefix (str, optional): Prefix for the JSON file to save results. Defaults to 'evaluation_results'.
            noise_level (float, optional): Noise level for the evaluation.Optional since the noise_level is here only for record purposes. Defaults to 0.00001.
            model_cut(bool or int): Either if the option model_last_layer has been choosen
        """
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
        self.input_tensor_size_init = tuple(abstract_domain['center'].size())
        self.model_cut =model_cut
        self.parallel_rel=parrallel_rel
        self.evaluator_rel= None
        if self.parallel_rel== True and self.renew_abstract_domain == True:
            raise Exception("Cannot renew abstract domain whith full parrallilsm ")
        if self.parallel_rel == True:
            self.evaluator_rel = SparseEvaluationParallel(self.abstract_domain['zonotope'],num_workers=self.num_workers,device=self.device)
            self.abstract_domain['zonotope'] =None


    def evaluate_model(self):
        """
        Evaluate the model layer by layer and save the results to a JSON file.

        Returns:
            dict: The updated abstract domain after evaluation.
        """
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
                "num_relevance_symbols": self.num_symbols,
                "input_tensor_size": self.input_tensor_size_init,
                "model_last_layer":self.model_cut,
                "Full parrallisation relevance": self.parallel_rel,
                "process_ended": False,
            },
            "layers": []
        }

        for name, details in self.model_unstacked.items():
            self.name = name
            self.details = details
            start_time = time.time()
            self.abstract_domain = process_layer(self.abstract_domain, self.name, self.details, self.num_workers, self.available_RAM, self.device, self.add_symbol,parallel=self.parallel_rel,evaluator_rel=self.evaluator_rel) 
            num_symbols=0
            nnz=0
            memory_gain_percentage=0
            input_tensor_size=0
            if  self.abstract_domain['zonotope'] is not None:
                if not self.parallel_rel:
                    num_symbols = self.abstract_domain['zonotope'].size(0)-self.num_symbols
                else:num_symbols = self.abstract_domain['zonotope'].size(0)
                nnz = self.abstract_domain['zonotope']._nnz()
            
                non_zero_percentage, memory_gain_percentage = sparse_tensor_stats(self.abstract_domain['zonotope'])
                input_tensor_size = tuple(self.abstract_domain['center'].size())
            end_time = time.time()
            computation_time = end_time - start_time

            

            layer_evaluation = {
                "layer_name": self.name,
                "layer_details": self.details,
                "input_tensor_size": input_tensor_size,
                "num_noise_symbols": num_symbols if num_symbols is not None else 0,
                "nnz": nnz,
                "memory_gain_percentage": float(memory_gain_percentage),
                "computation_time": computation_time,
                #"abstract_domain_center": self.abstract_domain['center'].tolist(),
                #"max_bounds": self.abstract_domain['sum'].tolist()
            }

            all_results["layers"].append(layer_evaluation)

            if self.renew_abstract_domain and self.abstract_domain['perfect_domain']:
                new_symbs = 2 * self.abstract_domain['sum'].to_dense()
                new_symbs_sparse = new_symbs.to_sparse_coo().coalesce()
                self.abstract_domain['zonotope'] = ZonoSparseGeneration().zono_from_tensor(new_symbs_sparse).coalesce()
                self.abstract_domain['trash'] = torch.zeros_like(new_symbs)


            table = [
                    ["Layer Name", layer_evaluation['layer_name']],
                    ["Input Tensor Size", layer_evaluation['input_tensor_size']],
                    ["Number of Noise Symbols", layer_evaluation['num_noise_symbols']],
                    ["Non-zero Elements (NNZ)", layer_evaluation['nnz']],
                    ["Memory Gain Percentage", f"{layer_evaluation['memory_gain_percentage']:.2f}%"],
                    ["Computation Time", f"{layer_evaluation['computation_time']:.6f} seconds"]
                    ]
            if self.verbose :
                print(tabulate(table, headers=["Field", "Value"], tablefmt="grid"))
            save_results_to_json(self.timestart, self.json_file_prefix, all_results)
        
        all_results['context']['process_ended'] = True
        save_results_to_json(self.timestart, self.json_file_prefix, all_results)
        
        return self.abstract_domain

  
"""
# Utilisation de la classe ModelEvaluator
# unstacked_model et abstract_domain doivent être définis avant cette utilisation
# evaluator = ModelEvaluator(unstacked_model, abstract_domain, verbose=True)
# abstract_domain = evaluator.evaluate_model()

"""