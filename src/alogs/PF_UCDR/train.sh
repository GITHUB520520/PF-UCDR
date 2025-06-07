# DomainNet
python3 main.py -data DomainNet -hd sketch -sd quickdraw -bs 50 -log 15 -lr 0.0001 -tp_N_CTX 16 -prompt 1 -debug_mode 0 --hpf_range 50 --hpf_alpha 0.4 --aug_alpha 1.0
python3 main.py -data DomainNet -hd quickdraw -sd sketch -bs 50 -log 15 -lr 0.0001 -tp_N_CTX 16 -prompt 1 -debug_mode 0 --hpf_range 50 --hpf_alpha 0.4 --aug_alpha 1.0
python3 main.py -data DomainNet -hd clipart -sd painting -bs 50 -log 15 -lr 0.0001 -tp_N_CTX 16 -prompt 1 -debug_mode 0 --hpf_range 50 --hpf_alpha 0.4 --aug_alpha 1.0
python3 main.py -data DomainNet -hd painting -sd infograph -bs 50 -log 15 -lr 0.0001 -tp_N_CTX 16 -prompt 1 -debug_mode 0 --hpf_range 50 --hpf_alpha 0.4 --aug_alpha 1.0
python3 main.py -data DomainNet -hd infograph -sd painting -bs 50 -log 15 -lr 0.0001 -tp_N_CTX 16 -prompt 1 -debug_mode 0 --hpf_range 50 --hpf_alpha 0.4 --aug_alpha 1.0

# Sketchy
python3 main.py -data Sketchy -bs 50 -log 15 -lr 0.0001 -tp_N_CTX 16 -prompt 1 -debug_mode 0 --hpf_range 50 --hpf_alpha 0.4 --aug_alpha 1.0

# TUBerlin
python3 main.py -data TUBerlin -bs 50 -log 15 -lr 0.0001 -tp_N_CTX 16 -prompt 1 -debug_mode 0 --hpf_range 50 --hpf_alpha 0.4 --aug_alpha 1.0