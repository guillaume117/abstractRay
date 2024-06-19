import argparse
import ray
import torch
from torch.sparse import FloatTensor
import sys
import copy
sys.path.append('app/src')
sys.path.append('./src')
from  sparse_evaluation_3 import SparseEvaluation  
from zono_sparse_gen import ZonoSparseGeneration
from abstract_relu import AbstractReLU


import torch.nn as nn


def main():
    conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3)
    epsilon_conv1 = copy.deepcopy(conv1)
    epsilon_conv1.bias.data = torch.zeros_like(epsilon_conv1.bias.data)
    conv2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3)
    func_conv2 = copy.deepcopy(conv2)
    func_conv2.bias.data = torch.zeros_like(func_conv2.bias.data)


    test= torch.randn(3,112,112)
    _,zono_from_test = ZonoSparseGeneration(test,0.001).total_zono()

    
    ray.init(include_dashboard=True)
    
    center = conv1(test)
    dim_chunck = dim_chunc(layer)
    evaluator = SparseEvaluation(zono_from_test,chunk_size =dim_chunk,function =epsilon_conv1,mask_epsilon=torch.ones_like(test))
    zono_conv_1,sum_abs =evaluator.evaluate_all_chunks(num_workers=9)
    trash = trash_conv1(trash_layer)
    center,trash_layer, mask_epsilon, new_sparse = AbstractReLU().abstract_relu(center,sum_abs,torch.zeros_like(center),start_index=3*56*56,trash_layer = trash_layer,add_symbol=True)
    
    print(new_sparse)

    center = conv2(center)

   
    evaluator = SparseEvaluation(zono_conv_1,chunk_size =2000,function =func_conv1)
    zono_conv_1,sum_abs =evaluator.evaluate_all_chunks(num_workers=9)
    print(zono_conv_1)

    ray.shutdown()


    

    


if __name__ == "__main__":
    main()
