defmodule Axon.Data.Utils do
  def unzip_cache_or_download(base_url, zip, data_path, opts \\ [unzip: true]) do
    unzip? = opts[:unzip] || false

    path = Path.join(data_path, zip)

    data =
      if File.exists?(path) do
        IO.puts("Using #{zip} from #{data_path}\n")
        File.read!(path)
      else
        IO.puts("Fetching #{zip} from #{base_url}\n")
        :inets.start()
        :ssl.start()

        {:ok, {_status, _response, data}} = :httpc.request(:get, {base_url ++ zip, []}, [], [])
        File.mkdir_p!(data_path)
        File.write!(path, data)

        data
      end

    :zlib.gunzip(data)
  end
end
