torchx? = System.get_env("USE_TORCHX") in ["1", "true"]
exla? = System.get_env("USE_EXLA") in ["1", "true"]

# Do not doctest if USE_EXLA or USE_TORCHX is set, because
# that will check for absolute equality and both will trigger
# failures
exclude_doctests = if torchx? or exla?, do: [test_type: :doctest], else: []

torchx_tests =
  if torchx? do
    [
      skip_torchx: :input_dilation,
      skip_torchx: :incompatible_implementations,
      skip_torchx: :window_dilations,
      skip_torchx: :padding
    ]
  else
    []
  end

ExUnit.start(exclude: exclude_doctests ++ torchx_tests ++ [skip_exla: exla?])
