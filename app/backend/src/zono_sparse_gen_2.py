import torch 


class ZonoSparseGeneration:
    def __init__(self):
        self.global_storage = {}
        self.global_storage['indices'] =[]
        self.global_storage['values'] = []
        pass
    def zono_from_tensor(self,noise_intensity):
        assert noise_intensity.size(0)==1,'First dimension size must be 1'
        noise_intensity = noise_intensity.to_sparse().coalesce()
        noise_intensity.indices()[0] = torch.arange(noise_intensity._nnz())
        size =(noise_intensity._nnz(),*noise_intensity.size()[1:])

        noise_intensity = torch.sparse_coo_tensor(noise_intensity.indices(),noise_intensity.values(),size = size)
        noise_intensity.coalesce()

        return(noise_intensity)

    def zono_from_noise_level_and_tensor(self,noise_level,tensor):
        
        noise_intensity = noise_level*torch.ones_like(tensor)
        zonotope = self.zono_from_tensor(noise_intensity=noise_intensity)
        return zonotope
