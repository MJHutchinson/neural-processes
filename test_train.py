from src.datagen.gpcurve import MaternGPCurvesReader, ProductRBFCurvesReader
from src.decoder import HeteroskedasticDecoder
from src.encoder import LatentEncoder, DeterministicEncoder
from src.aggregator import Attention
from src.neural_process import AttentiveNeuralProcess
from torch import optim
from src.train import train, train_spatiotemporal

MAX_CONTEXT_POINTS = 16
random_kernel_parameters = False
BATCH_SIZE = 16
hyperparameters = {
    "EPOCHS" : 10000,
    "PLOT_AFTER": 1000
}
# Sizes of the layers of the MLPs for the encoders and decoder
# The final output layer of the decoder outputs two values, one for the mean and
# one for the variance of the prediction at the target location
x_dim = 2
y_dim = 1
r_dim = 64
z_dim = 64
hid_dim = 128
num_hid = 3
detencoder = DeterministicEncoder(x_dim, y_dim, r_dim, hid_dim=hid_dim, num_hid=num_hid)
latencoder = LatentEncoder(x_dim, y_dim, z_dim, hid_dim=hid_dim, num_hid=num_hid)
decoder = HeteroskedasticDecoder(x_dim, r_dim + z_dim, y_dim, hid_dim=hid_dim, num_hid=2)
att = Attention('mlp','multihead', x_dim, r_dim, hid_dim=hid_dim, num_hid=num_hid)
model = AttentiveNeuralProcess(detencoder, att, latencoder, decoder, True)
optimizer = optim.Adam(model.parameters(), lr=0.001)

datagen = ProductRBFCurvesReader(BATCH_SIZE, MAX_CONTEXT_POINTS, random_kernel_parameters=random_kernel_parameters)

# Test dataset
datagen_test = ProductRBFCurvesReader(batch_size=1, max_num_context=MAX_CONTEXT_POINTS, testing=True, random_kernel_parameters=random_kernel_parameters)

y_target_mu, y_target_sigma, log_pred, kl_target_context, loss = train_spatiotemporal(model, hyperparameters, datagen, datagen_test, optimizer, save=True, experiment_name='anp_matern_gp')


print('FINISHED')
