import argparse
import ray
import torch
from torch.sparse import FloatTensor
import sys
sys.path.append('app/src')
sys.path.append('./src')
from  sparse_evaluation import SparseEvaluation  
import torch.nn as nn
def parse_arguments():
    parser = argparse.ArgumentParser(description="Sparse Evaluation with Ray")
    parser.add_argument('--num_chanel',type=int, required=False,help='num_chanel')
    parser.add_argument('--width',type=int, required=False,help='width')
    parser.add_argument('--height',type=int, required=False,help='height')
    parser.add_argument('--alpha',type=float, required=False,help='alpha')
    parser.add_argument('--sparse_tensor' ,type=FloatTensor,required = False, help = 'A sparse tensor for evaluation of noise symbols')
    parser.add_argument('--chunk_size', type=int, required=True, help='Chunk size for processing')
    parser.add_argument('--num_workers', type=int, required=True, help='Number of Ray workers')
    parser.add_argument('--mask',type=FloatTensor,required = False,help = 'mask for applying approximation of precedent non linear layer')
    parser.add_argument('--function',type = callable, required= False, help = 'A callable from nn.linear')

    return parser.parse_args()

def create_sparse_tensor(alpha,num_chanel, dim_H,dim_W):
    indices = []
    values = []
    dims=(num_chanel*dim_H*dim_W, num_chanel, dim_W, dim_H)
    num_elements = dims[0]


    for i in range(num_elements):

        dim_3 = i // (dim_W * dim_H)
        rem = i % (dim_W * dim_H)
        dim_1 = rem // dim_W
        dim_2 = rem % dim_H

    
        indices.append([i, dim_3, dim_1, dim_2])

        values.append(alpha)


    indices_tensor = torch.tensor(indices, dtype=torch.long).t()
    values_tensor = torch.tensor(values, dtype=torch.float)

    sparse_tensor = torch.sparse_coo_tensor(indices_tensor, values_tensor, size=dims).coalesce()

    return sparse_tensor

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
    evaluator = SparseEvaluation(x, args.chunk_size,function=function)
    global_sparse_tensor, function_sum = evaluator.evaluate_all_chunks(args.num_workers)

    ray.shutdown()
  


    # Print results
    print("Sparse Tensor:", global_sparse_tensor)
    print("Function Sum:", function_sum)

if __name__ == "__main__":
    main()
