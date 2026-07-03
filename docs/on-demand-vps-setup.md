# On-Demand Livestream — VPS Setup

Companion to `MediaMTXStreamer`'s `on_demand: true` mode. The edge (NAT'd,
outbound-only) polls a tiny **demand flag** on the VPS and pushes FFmpeg only
while a viewer is connected. MediaMTX's `runOnDemand`/`runOnUnDemand` hooks
(a) hold the WebRTC/WHEP viewer open during cold start and (b) write the flag
the edge polls.

## ⚠️ Image gotcha (verified)

The default `bluenviron/mediamtx:latest` image is built `FROM scratch` — **no
shell** — so `runOnDemand: sh -c '...'` silently does nothing (MediaMTX logs
"on demand command started" but nothing runs). Use
**`bluenviron/mediamtx:latest-ffmpeg`** (Alpine-based, has `sh`).

## docker-compose.yml — add the `demand` service + shared volume

```yaml
services:
  mediamtx:
    image: bluenviron/mediamtx:latest-ffmpeg   # NOT the scratch 'latest' — needs a shell for runOnDemand
    volumes:
      - ./mediamtx.yml:/mediamtx.yml:ro
      - demand:/demand                          # runOnDemand writes flags here
    # ...your existing ports/config...

  demand:
    image: nginx:alpine
    volumes:
      - demand:/usr/share/nginx/html/demand:ro  # serves the same flags, read-only
      - ./demand-nginx.conf:/etc/nginx/conf.d/default.conf:ro
    # no host port needed — reached over the compose network by your public proxy

volumes:
  demand:
```

`demand-nginx.conf`:

```nginx
server {
    listen 80;
    location /demand/ {
        default_type text/plain;
        add_header Cache-Control "no-store";    # never cache the flag
        # optional: allow only edge egress IPs, or require a shared-secret header
    }
}
```

## mediamtx.yml — one regex path covers all cameras

`$G1` capture becomes the flag filename (avoids the slash in `live/cam_sn_<id>`):

```yaml
paths:
  "~^live/cam_sn_(.+)$":
    source: publisher
    runOnDemand: sh -c 'echo on  > /demand/cam_sn_$G1'    # writes into the shared volume
    runOnUnDemand: sh -c 'echo off > /demand/cam_sn_$G1'
    runOnDemandStartTimeout: 40s   # holds the WHEP viewer open during cold start
    runOnDemandCloseAfter: 20s     # server-side grace before "no viewers" → writes off
```

## Public nginx — one proxy line

The edge polls `https://<vps>/demand/cam_sn_<id>` under your existing TLS.
Add one line to the public nginx so the flag shares the current hostname/cert
and the `demand` container stays internal to the compose network:

```nginx
location /demand/ { proxy_pass http://demand:80; }
```

## Edge config to match

The demand URL derives from the `mediamtx_ip` you already set, so normally
this is just the flag:

```yaml
advanced:
  livestream:
    mediamtx_ip: vps.example.com   # already set today; demand URL derives from this
    on_demand: true
    # optional overrides (defaults shown):
    # demand_poll_interval: 3.0
    # demand_grace_period: 10.0
    # demand_timeout: 5.0
    # demand_url: ""               # blank → https://vps.example.com/demand/cam_sn_<id>
```

If `mediamtx_ip` is a bare IP with no reverse proxy/TLS, set `demand_url`
explicitly (supports a `{camera_sn_id}` placeholder).

## Timing notes

- **`runOnDemandStartTimeout`** must exceed worst-case cold start seen by the
  held viewer = `poll_interval` (≤3 s) + FFmpeg `time.sleep(2.0)` init + RTSP
  connect (~1–2 s) + first keyframe (`-g fps` ⇒ ~1 s @30fps) ≈ 7–9 s typical.
  40 s leaves headroom; raise it if `poll_interval` is larger. Tune the two
  together.
- **Total teardown latency** ≈ `runOnDemandCloseAfter` (20 s) + up to
  `poll_interval` (3 s) + edge `demand_grace_period` (10 s) ≈ ~33 s of
  streaming after the last viewer leaves. Lower `grace_period` to trim, but
  keep it **≥ 5 s** so a stop→restart cycle never trips the streamer's 5 s
  rapid-restart guard.
- Missing flag file (path never demanded) → 404 → poll `ok=False` → demand
  unknown → edge stays idle. Correct behavior.

## Fail-safe policy (edge side, built in)

- Poll fails **while idle** → stay idle (if the VPS is unreachable nobody can
  be watching anyway).
- Poll fails **while live** → stay live (don't tear down a working stream
  over one flaky poll).
