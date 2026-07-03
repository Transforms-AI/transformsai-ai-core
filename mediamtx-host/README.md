# MediaMTX Host Setup

Deployment for the VPS that receives edge streams. Replaces the old
hand-run `docker run ... bluenviron/mediamtx:1` container with a compose
stack that also serves the **on-demand flag** used by
`MediaMTXStreamer(on_demand=True)` — the edge pushes FFmpeg only while at
least one viewer is watching, and stops shortly after the last viewer
leaves.

## Contents

| File | Purpose |
|---|---|
| `docker-compose.yml` | `mediamtx` + `demand` (flag server) + `demand-init` (stale-flag cleanup) |
| `mediamtx.yml` | Full MediaMTX config (previous VPS config + the on-demand path hooks) |
| `demand-nginx.conf` | nginx config for the flag server |

## How it works

The edge is NAT'd (outbound-only), so the VPS can't dial in. Instead:

1. A viewer opens the WebRTC/WHEP URL for `live/cam_sn_<id>` while nobody
   is publishing. MediaMTX's `runOnDemand` hook fires: it writes `on` to
   `/demand/cam_sn_<id>` in a shared Docker volume, and MediaMTX **holds
   the viewer's connection open** for up to `runOnDemandStartTimeout`
   (40 s) instead of returning 404.
2. The edge polls `https://<host>/demand/cam_sn_<id>` every
   `demand_poll_interval` (3 s), sees `on`, and starts its FFmpeg push.
   The held viewer gets video — typically within ~7–9 s of clicking play.
3. When the last viewer leaves and `runOnDemandCloseAfter` (20 s) passes,
   `runOnUnDemand` writes `off`. The edge sees it, waits its own
   `demand_grace_period` (10 s), and stops FFmpeg. Total teardown ≈ 33 s
   after the last viewer; idle upstream bandwidth is zero.

Fail-safe on the edge: an unreachable/unrecognized flag never changes
state — idle stays idle, live stays live.

## ⚠️ Image requirement

The stack uses `bluenviron/mediamtx:latest-ffmpeg`, **not** the previous
`bluenviron/mediamtx:1`. The default image is built `FROM scratch` with no
shell, so the `sh -c 'echo on > ...'` hooks silently do nothing (MediaMTX
logs "on demand command started" but nothing runs). The `-ffmpeg` variant
is Alpine-based and has `sh`. Everything else about the server behaves the
same.

## Deploying (migration from the old container)

```bash
# on the VPS, from this directory
sudo docker rm -f mediamtx          # stop the old hand-run container
sudo docker compose up -d
sudo docker compose logs -f mediamtx
```

The compose file carries over everything from the old run command: the
port mappings, `MTX_RTSPTRANSPORTS=tcp`,
`MTX_WEBRTCADDITIONALHOSTS=161.97.126.245,mediamtx.transformsai.com`, and
the 1 GB / 2 CPU resource limits. `demand-init` wipes leftover flags on
every `up` so a stale `on` from before a reboot can't make an edge stream
to nobody.

## Exposing the demand flag

The `demand` nginx container binds to `127.0.0.1:8880` on the host. Add
one location to the public nginx that already terminates TLS for
`mediamtx.transformsai.com`:

```nginx
location /demand/ { proxy_pass http://127.0.0.1:8880; }
```

That gives the edges `https://mediamtx.transformsai.com/demand/cam_sn_<id>`
under the existing hostname/cert — which is exactly the URL the library
derives from `mediamtx_ip`, so no `demand_url` override is needed.

(No public proxy? Change the port mapping to `"8880:80"` and set
`demand_url: "http://<host>:8880/demand/cam_sn_{camera_sn_id}"` on the
edge instead.)

## Edge config to match

```yaml
advanced:
  livestream:
    mediamtx_ip: mediamtx.transformsai.com   # demand URL derives from this
    on_demand: true
    # optional overrides (defaults shown):
    # demand_poll_interval: 3.0
    # demand_grace_period: 10.0   # keep >= 5s (streamer's rapid-restart guard)
    # demand_timeout: 5.0
    # demand_url: ""
```

`on_demand: false` (the default) keeps today's always-on behavior; nothing
else changes.

## Tuning

- `runOnDemandStartTimeout` (40 s) must exceed the edge's worst-case cold
  start: `demand_poll_interval` + FFmpeg init (2 s) + RTSP connect
  (~1–2 s) + first keyframe (~1 s). Raise it if you raise the poll
  interval — tune the two together.
- Teardown latency ≈ `runOnDemandCloseAfter` + `demand_poll_interval` +
  `demand_grace_period`. Lower `runOnDemandCloseAfter` or `grace_period`
  to trim, but keep `grace_period` ≥ 5 s.

## Verifying

```bash
# 1) Flag starts absent -> edge stays idle (404 is the correct idle answer)
curl -i https://mediamtx.transformsai.com/demand/cam_sn_<id>

# 2) Open the viewer: https://mediamtx.transformsai.com/live/cam_sn_<id>/
#    The page should hold (not 404) and show video within ~10s.
curl https://mediamtx.transformsai.com/demand/cam_sn_<id>   # now "on"

# 3) Close the viewer; after ~20s the flag flips
curl https://mediamtx.transformsai.com/demand/cam_sn_<id>   # now "off"
#    and the edge's FFmpeg exits ~10s later (check .core-streamer-logs on the edge)

# Hook execution check (should appear when a viewer connects):
sudo docker compose logs mediamtx | grep -i "on demand"
# If you see "on demand command started" but the flag file never appears,
# you are on the shell-less scratch image — use latest-ffmpeg.
```

## Notes

- `mediamtx.yml` contains the Control API credentials (`transformsai`
  user). This repo is internal, but consider moving the password to an env
  var (`MTX_AUTHINTERNALUSERS_1_PASS`) if the repo's audience widens.
- The demand endpoint is public read-only. The flags leak nothing beyond
  "someone is watching camera X"; restrict by edge egress IP or a
  shared-secret header in `demand-nginx.conf` if that matters.
