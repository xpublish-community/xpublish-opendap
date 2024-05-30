# Testing & Debugging

With the complexities of OpenDAP, there are some datasets that don't behave as expected.

To figure out how to deal with those, there is a `docker-compose.yml` file to allow standing up a [`Hyrax` OpenDAP server](https://github.com/OPENDAP/hyrax-docker/tree/master) to compare responses too.

- To add data to the server, include the NetCDF in the `tests/data` directory.
- To launch the server, `docker compose up`. `ctrl-c` to stop, and `docker compose down` to shut it down.
- To access the server, visit [http://127.0.0.1:8080/opendap/](http://127.0.0.1:8080/opendap/).
- To run an Xpublish server (the same one that is run by tests), `python server.py`, will serve the same datasets and a few others.

It also can be useful to use a proxy to compare traffic between Xpublish and Hyrax.

[Mitmproxy](https://mitmproxy.org/) is one that can work. If started with `mitmweb -p 8081 --web-port 8082` it will launch a web interface to compare flows and proxy traffic on `8081`.

`NetCDF4` will use the `http_proxy` environment variable, so setting `os.environ["http_proxy"] = "http://127.0.0.1:8081/"` before `xr.open_dataset` to either server will cause it's traffic to be captured.
