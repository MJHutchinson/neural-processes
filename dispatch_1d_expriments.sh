python toy_gp_train.py -m CNP --GP-type RBF
python toy_gp_train.py -m ANP_CNP --GP-type RBF
python toy_gp_train.py -m ANP --GP-type RBF
python toy_gp_train.py -m NP --GP-type RBF
python toy_gp_train.py -m ConvCNP --GP-type RBF

python toy_gp_train.py -m CNP --GP-type Matern
python toy_gp_train.py -m ANP_CNP --GP-type Matern
python toy_gp_train.py -m ANP --GP-type Matern
python toy_gp_train.py -m NP --GP-type Matern
python toy_gp_train.py -m ConvCNP --GP-type Matern

python toy_gp_train.py -m ConvCNPXL --GP-type RBF
python toy_gp_train.py -m ConvCNPXL --GP-type Matern