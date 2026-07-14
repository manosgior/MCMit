Module for running a multi-board job server. Clients can submit batches of compiled executables to this server using the normal [CircuitRunnerClient](../rpc_client#qubic.rpc_client.CircuitRunnerClient). These executables can span multiple boards, as resolved by the [ChannelConfig]() file. For details on configuring and running the job server, see [(wiki page coming soon!)](). Individual boards will receive partial executables and synchronized triggers via the normal [soc_rpc_server](../soc_rpc_server).

::: qubic.job_rpc_server
