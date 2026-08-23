torchx? = System.get_env("USE_TORCHX") in ["1", "true"]
exla? = System.get_env("USE_EXLA") in ["1", "true"]

# Do not doctest if USE_EXLA or USE_TORCHX is set, because
# that will check for absolute equality and both will trigger
# failures
exclude_doctests = if torchx? or exla?, do: [test_type: :doctest], else: []

# Tests tagged `exla_only` assert on behaviour only EXLA implements, such as
# buffer donation, so they only run when USE_EXLA is set.
ExUnit.start(
  exclude:
    exclude_doctests ++
      [skip_torchx: torchx?, skip_exla: exla?, exla_only: not exla?, integration: true]
)
