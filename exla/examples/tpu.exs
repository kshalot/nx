defmodule Softmax do
  import Nx.Defn
  @default_defn_compiler {EXLA, client: :tpu, max_float_type: {:f, 32}}

  defn softmax(t), do: Nx.exp(t) / Nx.sum(Nx.exp(t))
end

IO.inspect Softmax.softmax(Nx.tensor([1, 2, 3, 4], type: {:f, 32}))
