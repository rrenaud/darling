set -e -x
torchrun --nproc-per-node=4 --standalone tests/special_distributed/test_tensor_dict.py