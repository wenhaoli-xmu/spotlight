import triton


def block_n_config():
    return [
        triton.Config(kwargs={"BLOCK_N": 32}),
        triton.Config(kwargs={"BLOCK_N": 64}),
        triton.Config(kwargs={"BLOCK_N": 128}),
        triton.Config(kwargs={"BLOCK_N": 256}),
    ]