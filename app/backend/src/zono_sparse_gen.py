import torch 
import matplotlib.pyplot as plt

class ZonoSparseGeneration:
    def __init__(self):
        self.global_storage = {}
        self.global_storage['indices'] =[]
        self.global_storage['values'] = []
        pass



    def zono_from_input_noise_level_and_mask(self,tensor_input, x_min,x_max,y_min,y_max, noise_level):
        output = noise_level*torch.zeros_like(tensor_input)
        output[:,:,x_min:x_max,y_min:y_max] = noise_level 
        zonotope = self.zono_from_tensor(output)
        return zonotope.coalesce()




    def zono_from_tensor(self,noise_intensity,start_index =0):
        assert noise_intensity.size(0)==1,'First dimension size must be 1'
        noise_intensity = noise_intensity.to_sparse().coalesce()
        noise_intensity.indices()[0] = torch.arange(noise_intensity._nnz())+start_index
        size =(noise_intensity._nnz()+start_index,*noise_intensity.size()[1:])

        zonotope = torch.sparse_coo_tensor(noise_intensity.indices(),noise_intensity.values(),size = size)
       

        return zonotope.coalesce() 

    def zono_from_noise_level_and_tensor(self,noise_level,tensor):
        
        noise_intensity = noise_level*torch.ones_like(tensor)
        zonotope = self.zono_from_tensor(noise_intensity=noise_intensity)
        return zonotope.coalesce()
    

if __name__ == "__main__":   
    test=ZonoSparseGeneration().zono_from_input_noise_level_and_mask(torch.randn(1,3,52,52),10,15,10,15,1)
    print(test)
    plt.imshow(torch.sum(test.coalesce(),dim=0).to_dense().permute(1,2,0))
    plt.show()