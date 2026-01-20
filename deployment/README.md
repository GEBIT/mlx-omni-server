# How to deploy mlx-omni-server as a service

First, make sure that `mlx-omni-server` is installed properly according to the parent README! Make sure you ran `uv sync` on the most recent commit.

1. Adjust paths in start_server.sh if necessary and make sure it is executable (`chmod +x`). Test it by executing it.
2. `cp com.yourname.mlx_omni_server.plist ~/Library/LaunchAgents/com.yourname.mlx_omni_server.plist`
3. Adjust paths in `~/Library/LaunchAgents/com.yourname.mlx_omni_server.plist` to fit your user and the location of this repo.
4. Run `launchctl load ~/Library/LaunchAgents/com.yourname.mlx_omni_server.plist` to load (and launch) to service.
5. Check that the service is running: `launchctl list | grep mlx_omni_server`

Stop the service:

    launchctl unload ~/Library/LaunchAgents/com.yourname.mlx_omni_server.plist

Start the service:

    launchctl load ~/Library/LaunchAgents/com.yourname.mlx_omni_server.plist

Thanks to the `KeepAlive` setting in the plist file, the service will automatically restart after crashing. It will also start automatically when the current user logs in (after a reboot).

Logs are located at:

    /tmp/mlx_server.log
    /tmp/mlx_server_error.log
