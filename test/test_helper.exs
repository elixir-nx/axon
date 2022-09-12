torchx? = System.get_env("USE_TORCHX") in ["1", "true"]
exla? = System.get_env("USE_EXLA") in ["1", "true"]

ExUnit.start(exclude: [skip_torchx: torchx?, skip_exla: exla?])
