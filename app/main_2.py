import argparse
import ray
import torch
from torch.sparse import FloatTensor
import sys
sys.path.append('app/src')
sys.path.append('./src')
from  sparse_evaluation import SparseEvaluation  
from zono_sparse_gen import ZonoSparseGeneration


import torch.nn as nn

def parse_arguments():
    parser = argparse.ArgumentParser(description="Sparse Evaluation with Ray")
    parser.add_argument('--num_chanel', type=int, default=3, help='Number of channels')
    parser.add_argument('--width', type=int, default=112, help='Width of the input')
    parser.add_argument('--height', type=int, default=112, help='Height of the input')
    parser.add_argument('--chunk_size', type=int, default=500, help='Chunk size')
    parser.add_argument('--num_worker', type=int, default=10, help='Number of workers')
    parser.add_argument('--alpha', type=float, default=0.0001, help='Alpha value')



    parser.add_argument('--mask',type=FloatTensor,required = False,help = 'mask for applying approximation of precedent non linear layer')
    parser.add_argument('--function',type = callable, required= False, help = 'A callable from nn.linear')

    return parser.parse_args()



def main():
    args = parse_arguments()
   
    num_chanel = args.num_chanel
    alpha = args.alpha
    dim_H = args.height
    dim_W = args.width
    x=create_sparse_tensor(alpha,num_chanel, dim_H,dim_W).coalesce()

    conv = nn.Conv2d(in_channels=num_chanel, out_channels=64,kernel_size=3)
    conv.bias.data = torch.zeros_like(conv.bias.data)
    function = lambda x : conv(x)

    

    # Initialize Ray
    ray.init(include_dashboard=True)

    # Initialize and run the evaluation
    evaluator = SparseEvaluation(x, args.chunk_size,mask_coef = None ,function=function)
    global_sparse_tensor, function_sum = evaluator.evaluate_all_chunks(args.num_worker)

    ray.shutdown()
  


    # Print results
    print("Sparse Tensor:", global_sparse_tensor)
   

if __name__ == "__main__":
    main()
