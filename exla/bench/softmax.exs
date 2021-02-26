size = 1_000_000
t16 = Nx.tensor(for(_ <- 1..size, do: :rand.uniform()), type: {:bf, 16})
t32 = Nx.tensor(for(_ <- 1..size, do: :rand.uniform()), type: {:f, 32})

defmodule Softmax do
  import Nx.Defn

  # This runs on Elixir
  defn softmax(n), do: Nx.exp(n) / Nx.sum(Nx.exp(n))

  # This is JIT+host compiled. By default EXLA sets max_float_type to 32,
  # so we override it here and limit it on the tensor input
  @defn_compiler {EXLA, max_float_type: {:f, 64}}
  defn host(n), do: softmax(n)

  # This is JIT+tpu compiled
  @defn_compiler {EXLA, client: :tpu, max_float_type: {:f, 32}}
  defn tpu(n), do: softmax(n)
end

IO.inspect(Softmax.softmax(t32))
IO.inspect(Softmax.host(t32))

benches = %{
  "elixir bf16" => fn -> Softmax.softmax(t16) end,
  "elixir f32" => fn -> Softmax.softmax(t32) end,
  "xla cpu bf16" => fn -> Softmax.host(t16) end,
  "xla cpu f32" => fn -> Softmax.host(t32) end,
  "xla tpu bf16" => {fn -> Softmax.tpu(t16) end, after_each: fn _ -> :erlang.garbage_collect() end},
  "xla tpu f32" => {fn -> Softmax.tpu(t32) end, after_each: fn _ -> :erlang.garbage_collect() end}
}

Benchee.run(
  benches,
  time: 10,
  memory_time: 2
)
