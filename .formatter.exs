locals_without_parens = [
  defn: 2,
  defnp: 2
]

[
  plugins: [],
  locals_without_parens: locals_without_parens,
  export: [locals_without_parens: locals_without_parens],
  inputs: ["{mix,.formatter}.exs", "{config,lib,test}/**/*.{ex,exs}"],
  import_deps: [:nx]
]
