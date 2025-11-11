# How to deploy mlx-omni-server as a service

1. Adjust paths in start_server.sh if necessary and make sure it is executable (`chmod +x`).
2. Copy `de.gebit.mlx_omni_server.plist` to `~/Library/LaunchAgents/de.gebit.mlx_omni_server.plist`.
3. Run `launchctl load ~/Library/LaunchAgents/de.gebit.mlx_omni_server.plist` to load (and launch) to service.
4. Check that the service is running: `launchctl list | grep mlx_omni_server`

Stop the service:

    launchctl unload ~/Library/LaunchAgents/de.gebit.mlx_omni_server.plist

Start the service:

    launchctl load ~/Library/LaunchAgents/de.gebit.mlx_omni_server.plist

Thanks to the `KeepAlive` setting in the plist file, the service will automatically restart after crashing. It will also start automatically when the current user logs in (after a reboot).

Logs are located at:

    /tmp/mlx_server.log
    /tmp/mlx_server_error.log