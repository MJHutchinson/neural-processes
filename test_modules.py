from torch.distributions import Normal

from src.decoder import *
from src.encoder import *
from src.aggregator import *
from src.neural_process import *

x_dim = 2
z_dim = 3
r_dim = 4
rep_dim = z_dim + r_dim
y_dim = 5
h_dim = 10
num_h = 3

batches = 100
num_target = 10
num_context = 20

X_context = torch.Tensor(batches, num_context, x_dim).normal_()
Y_context = torch.Tensor(batches, num_context, y_dim).normal_()
X_target = torch.Tensor(batches, num_target, x_dim).normal_()
Y_target = torch.Tensor(batches, num_target, y_dim).normal_()

EncD = DeterministicEncoder(x_dim, y_dim, r_dim, h_dim, num_h)

R_context = EncD(X_context, Y_context)
R_target = EncD(X_target, Y_target)

print(f'Deterministic encoder output shape: {R_context.size()}, should be {(batches, num_context, r_dim)}')
print(f'Deterministic encoder output shape: {R_target.size()}, should be {(batches, num_target, r_dim)}')

AttenIden = Attention('identity', 'uniform', x_dim, r_dim)

R_context_atten_iden = AttenIden(X_context, X_target, R_context)
R_target_atten_iden = AttenIden(X_target, X_target, R_target)

print(f'Identity attention output shape: {R_context_atten_iden.size()}, should be {(batches, num_context, r_dim)}')
print(f'Identity attention output shape: {R_target_atten_iden.size()}, should be {(batches, num_target, r_dim)}')

EncL = LatentEncoder(x_dim, y_dim, z_dim, h_dim, num_h)

Z_context = EncL(X_context, Y_context)
Z_target = EncL(X_target, Y_target)

Z_context_sample = Normal(Z_context[0], Z_context[1]).sample().unsqueeze(dim=1).expand(-1, num_context, -1)
Z_target_sample = Normal(Z_target[0], Z_target[1]).sample().unsqueeze(dim=1).expand(-1, num_target, -1)

print(f'Latent encoder output shape: {Z_context_sample.size()}, should be {(batches, num_context, z_dim)}')
print(f'Latent encoder output shape: {Z_target_sample.size()}, should be {(batches, num_target, z_dim)}')

Rep_context = torch.cat((R_context, Z_context_sample), dim=-1)
Rep_target = torch.cat((R_target, Z_target_sample), dim=-1)

print(f'Rep shape: {Rep_context.size()}, should be {(batches, num_context, rep_dim)}')
print(f'Rep output shape: {Rep_target.size()}, should be {(batches, num_target, rep_dim)}')

HetD = HeteroskedasticDecoder(x_dim, rep_dim, y_dim, h_dim, num_h)
HomD = HomoskedasticDecoder(x_dim, rep_dim, y_dim, h_dim, num_h, 1)

Y_pred_context_het = HetD(X_context, Rep_context)
Y_pred_target_het = HetD(X_target, Rep_target)

print(f'Het decoder output shape: {Y_pred_context_het[0].size(), Y_pred_context_het[1].size()}, should be {(batches, num_context, y_dim)}')
print(f'Het decoder output shape: {Y_pred_target_het[0].size(), Y_pred_target_het[1].size()}, should be {(batches, num_target, y_dim)}')

Y_pred_context_Hom = HomD(X_context, Rep_context)
Y_pred_target_Hom = HomD(X_target, Rep_target)

print(f'Hom decoder output shape: {Y_pred_context_Hom[0].size(), Y_pred_context_Hom[1].size()}, should be {(batches, num_context, y_dim)}')
print(f'Hom decoder output shape: {Y_pred_target_Hom[0].size(), Y_pred_target_Hom[1].size()}, should be {(batches, num_target, y_dim)}')

Atten_ANP = Attention('mlp', 'multihead', x_dim, r_dim, num_heads=r_dim)

ANP_NP = AttentiveNeuralProcess(x_dim, y_dim, r_dim, z_dim, EncD, AttenIden, EncL, HetD, True)
ANP_ANP = AttentiveNeuralProcess(x_dim, y_dim, r_dim, z_dim, EncD, Atten_ANP, EncL, HomD, True)

Y_pred_ANP_Iden = ANP_NP(X_context, Y_context, X_target, Y_target)
Y_pred_ANP_ANP = ANP_ANP(X_context, Y_context, X_target, Y_target)

print(f'ANP_Iden output: {[r.size() for r in Y_pred_ANP_Iden]}')
print(f'ANP_ANP output: {[r.size() for r in Y_pred_ANP_ANP]}')