import sys,torch
sys.path.insert(0,'/data/wujiaju/n10_migration_20260923/code')
import reasoning_loop.paper2027_d8l6_s1 as old
class DenseAffine(torch.nn.Module):
 def __init__(self,dimension,rank=None):
  super().__init__();self.weight=torch.nn.Parameter(torch.eye(dimension));self.bias=torch.nn.Parameter(torch.zeros(dimension))
 def forward(self,x):return x.float()@self.weight+self.bias
old.DiagonalLowRankGraphController=DenseAffine
original=old.payload
def payload(*args,**kw):
 d=original(*args,**kw);d.update(kind='dense_affine_matched_v1',parameterization='hW+b',parameter_count=65792);return d
old.payload=payload
torch.set_num_threads(4)
def load_controller(path,checkpoint,device):
 item=torch.load(path,map_location='cpu',weights_only=False)
 assert item['kind']=='dense_affine_matched_v1' and item['checkpoint_sha256']==old.sha256(checkpoint)
 j=DenseAffine(256).to(device);j.load_state_dict(item['controller_state_dict']);j.eval();return j,item
old.load_controller=load_controller
if __name__=='__main__':
 args=old.parse_args()
 old.train(args) if args.action=='train' else old.evaluate(args)

