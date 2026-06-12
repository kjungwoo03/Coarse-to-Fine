
import torch.nn as nn
import torch 

def ste_round(x):
    return torch.round(x) - x.detach() + x


class ChannelMask(nn.Module):

    def __init__(self,mask_policy):
        super().__init__()

        self.mask_policy = mask_policy 

    
    
    def ProgMask(self,scale,pr):
        """
        in this scenario scale is a list of blocks  
        """
        mask = []
        number_blocks = len(scale) 
        pr = 10 if pr > 10 else pr
        pr = pr*0.1
        pr_bis = 1.0 - pr
        for bl in range(number_blocks):
            #print("block shape: ",scale[bl].shape)
            block = scale[bl]
            bs, ch, w,h = block.shape
            if pr >= 1:
                mask.append(torch.ones_like(block).to(block.device).squeeze(0)) 
                continue
            res = torch.zeros_like(block).to(block.device).squeeze(0)
            if pr == 0: 
                mask.append(res) 
                continue 
            for j in range(bs): # it is one 
                scale_b = block[j,:,:,:] 
                scale_b = scale_b.ravel()
                quantile = torch.quantile(scale_b, pr_bis)
                res_b = scale_b >= quantile #if "inverse" not in mask_pol else  scale_b <= quantile
                res_b = res_b.reshape(ch,w,h).float()
                res_b = res_b.to(block.device)
                mask.append(res_b) 
        
        #mask is a list of [32,]
        res = torch.stack(mask)             
        return res           
            

    def delta_mask(self, scale,  pr_bar, pr):
        shapes = scale.shape
        bs, ch, w,h = shapes
        assert pr_bar <= pr
        if pr_bar >= 10:
            return torch.ones_like(scale).to(scale.device)
        elif pr_bar == 0:
            return torch.zeros_like(scale).to(scale.device) 
        assert scale is not None 
        pr = 10 if pr > 10 else pr
        pr = pr*0.1
        pr = 1.0 - pr

        pr_bar = 10 if pr_bar > 10 else pr_bar
        pr_bar = pr_bar*0.1
        pr_bar = 1.0 - pr_bar
        res = torch.zeros_like(scale).to(scale.device)   
        for j in range(bs):
            scale_b = scale[j]#scale[j,:,:,:]
            scale_b = scale_b.ravel()
            quantile_bar = torch.quantile(scale_b, pr_bar)
            quantile = torch.quantile(scale_b, pr)
            res_b = quantile_bar >= scale_b > quantile 
            res_b = res_b.reshape(ch,w,h)
            res_b = res_b.to(scale.device)
            res[j] = res_b   
        return res.to(scale.device)          


    def apply_noise(self, mask, training):
            if training:
                mask = ste_round(mask)
            else:
                mask = torch.round(mask)
            return mask


    def forward(self,
                scale,  
                pr = 0, 
                mask_pol = "point-based-std",
                ravel = False,
                cust_map = None):

        if cust_map is not None:

            if pr >= 10:
                return torch.ones_like(cust_map).to(scale.device)
            elif pr == 0:
                return torch.zeros_like(cust_map).to(scale.device)  
            #cust_map = cust_map.unsqueeze(0).to(cust_map.device)
            shapes = cust_map.shape

         
            pr = 10 if pr > 10 else pr
            pr = pr*0.1
            pr_bis = 1.0 - pr
            res = torch.zeros_like(cust_map).to(scale.device)            

            for j in range(bs):
                cust_map_b = cust_map[j,:,:,:]
                cust_map_b = cust_map_b.ravel()
                quantile = torch.quantile(cust_map_b, pr_bis)
                res_b = cust_map_b >= quantile #if "inverse" not in mask_pol else  scale_b <= quantile
                res_b = res_b.reshape(ch,w,h)
                res_b = res_b.to(scale.device)
                res[j] = res_b

            #print("struttura della maschera for: ",pr,": ", torch.unique(res,return_counts = True))
            return res.to(scale.device) 
        
        if mask_pol is None:
            mask_pol = self.mask_policy

        shapes = scale.shape
        if ravel is False:
            bs, ch, w,h = shapes
        else:
            bs,d = shapes

        if mask_pol == "point-based-std":
            if pr >= 10:
                return torch.ones_like(scale).to(scale.device)
            elif pr == 0:
                return torch.zeros_like(scale).to(scale.device)
            assert scale is not None 
            pr = 10 if pr > 10 else pr
            pr = pr*0.1
            pr_bis = 1.0 - pr
            res = torch.zeros_like(scale).to(scale.device)
            for j in range(bs):
                scale_b = scale[j,:,:,:] if ravel is False else scale[j]
                scale_b = scale_b.ravel()
                quantile = torch.quantile(scale_b, pr_bis)
                res_b = scale_b >= quantile #if "inverse" not in mask_pol else  scale_b <= quantile
                res_b = res_b.reshape(ch,w,h) if ravel is False else res_b.reshape(d)
                res_b = res_b.to(scale.device)
                res[j] = res_b
            return res.float().reshape(bs,ch,w,h).to(torch.float).to(scale.device) if ravel is False \
                    else res.float().reshape(bs,d).to(torch.float).to(scale.device)
        elif mask_pol == "two-levels":
            return torch.zeros_like(scale).to(scale.device) if pr == 0 else torch.ones_like(scale).to(scale.device)

        else:
            raise NotImplementedError()