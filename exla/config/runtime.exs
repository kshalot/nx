import Config

target = System.get_env("EXLA_TARGET", "host")

config :exla, :clients, default: [platform: :host],
  tpu: [platform: :tpu]

config :logger, :console,
  format: "\n$time [$level] $metadata $levelpad$message\n",
  metadata: [:domain, :file, :line]
