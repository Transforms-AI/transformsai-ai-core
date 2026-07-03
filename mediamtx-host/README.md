# MediaMTX Host Setup

Deployment for the VPS that receives edge streams. Replaces a hand-run
`docker run ... bluenviron/mediamtx` container with a compose stack that
also serves the **on-demand flag** used by
`MediaMTXStreamer(on_demand=True)` — the edge pushes FFmpeg only while at
least one viewer is watching, and stops shortly after the last viewer
leaves.

## Contents

| File | Purpose |
|---|---|
| `docker-compose.yml` | `mediamtx` + `demand` (flag server) + `demand-init` (stale-flag cleanup) |
| `mediamtx.yml` | Full MediaMTX config (server settings + the on-demand path hooks) |
| `demand-nginx.conf` | nginx config for the flag server container |

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

The stack uses `bluenviron/mediamtx:latest-ffmpeg`, **not** the default
`bluenviron/mediamtx:latest`/`:1`. The default image is built
`FROM scratch` with no shell, so the `sh -c 'echo on > ...'` hooks
silently do nothing (MediaMTX logs "on demand command started" but
nothing runs). The `-ffmpeg` variant is Alpine-based and has `sh`.
Everything else about the server behaves the same.

---

## Full setup from scratch

Assumes a fresh Ubuntu VPS with a public IP and a domain you control.
Concrete values below use the production host
(`mediamtx.transformsai.com`, IP `161.97.126.245`) — substitute your own.

### 1. DNS

Point an A record at the VPS: `mediamtx.transformsai.com → 161.97.126.245`.

### 2. Install Docker and nginx

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo apt update && sudo apt install -y nginx certbot python3-certbot-nginx
```

### 3. Get this folder onto the VPS

```bash
git clone https://github.com/Transforms-AI/transformsai-ai-core.git
cd transformsai-ai-core/mediamtx-host
```

(Or `scp -r mediamtx-host/ user@vps:~/` — the folder is self-contained.)

If your domain/IP differ, edit `MTX_WEBRTCADDITIONALHOSTS` in
`docker-compose.yml`.

### 4. Start the stack

```bash
# if migrating from an old hand-run container, remove it first:
sudo docker rm -f mediamtx 2>/dev/null

sudo docker compose up -d
sudo docker compose logs -f mediamtx    # ctrl-c to detach
```

This starts:
- **mediamtx** — RTSP :8554 (edge publishes), WebRTC :8889 + :8189/udp
  (viewers), Control API :9997, with the on-demand hooks active.
- **demand** — flag server on `127.0.0.1:8880` (host-local only; the
  public nginx proxies to it in the next step).
- **demand-init** — one-shot that wipes stale flags on every `up`, so a
  leftover `on` from before a reboot can't make an edge stream to nobody.

### 5. TLS certificate

```bash
sudo certbot certonly --nginx -d mediamtx.transformsai.com
```

### 6. Public nginx site

Create `/etc/nginx/sites-available/mediamtx.transformsai.com` with the
**complete file** below. It does two things: proxies everything to
MediaMTX's WebRTC server (:8889) for viewing, and proxies `/demand/` to
the flag container (:8880) for the edges. The `location /demand/` block
must come as a sibling of `location /` — nginx picks the longest prefix
match, so `/demand/...` requests go to the flag server and everything
else to MediaMTX.

```nginx
server {
    listen 443 ssl;
    server_name mediamtx.transformsai.com;

    # Demand flags for on-demand edge publishing
    # (served by the mediamtx-demand container, bound to 127.0.0.1:8880)
    location /demand/ {
        proxy_pass http://127.0.0.1:8880;
    }

    location / {
        # Proxy to MediaMTX WebRTC HTTP port (default 8889)
        proxy_pass http://127.0.0.1:8889;

        # Essential headers for WebRTC/WebSocket signaling
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;

        # Pass client IP to MediaMTX
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    ssl_certificate /etc/letsencrypt/live/mediamtx.transformsai.com/fullchain.pem; # managed by Certbot
    ssl_certificate_key /etc/letsencrypt/live/mediamtx.transformsai.com/privkey.pem; # managed by Certbot
}
```

Enable and reload:

```bash
sudo ln -sf /etc/nginx/sites-available/mediamtx.transformsai.com /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

> Getting a **302 redirect** with `Access-Control-Allow-*` headers from
> `curl https://<host>/demand/cam_sn_x`? That's MediaMTX answering — the
> `/demand/` location is missing or in the wrong server block, so the
> request fell through to :8889.

### 7. Open the firewall

Besides 443, edges and viewers need these reachable:

```bash
sudo ufw allow 443/tcp    # WebRTC signaling + demand flags (via nginx)
sudo ufw allow 8554/tcp   # RTSP publish from edges
sudo ufw allow 8189/udp   # WebRTC media (ICE)
```

(Ports 8880 and 9997 stay private; 1935/8888/8890 are only needed if you
enable RTMP/HLS/SRT in `mediamtx.yml`.)

### 8. Configure the edges

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

`on_demand: false` (the default) keeps always-on behavior; nothing else
changes. Viewers watch at
`https://mediamtx.transformsai.com/live/cam_sn_<id>/`.

### 9. Verify

**VPS half (no edge needed):**

```bash
# flag absent -> 404 is the correct idle answer
curl -i https://mediamtx.transformsai.com/demand/cam_sn_test1
```

Open `https://mediamtx.transformsai.com/live/cam_sn_test1/` in a browser —
the page should **hold/spin** (that's `runOnDemandStartTimeout` working),
not instantly fail. While it spins:

```bash
curl https://mediamtx.transformsai.com/demand/cam_sn_test1   # -> on
# close the tab, wait ~25s (runOnDemandCloseAfter)
curl https://mediamtx.transformsai.com/demand/cam_sn_test1   # -> off
```

**Full loop (with an edge):** run a quick publisher on any machine with
the library:

```python
# demo_on_demand.py  ->  uv run python demo_on_demand.py
import time, cv2, numpy as np
from transformsai_ai_core import MediaMTXStreamer

s = MediaMTXStreamer("mediamtx.transformsai.com", 8554, "test1",
                     fps=15, frame_width=640, frame_height=480,
                     on_demand=True)
s.start_streaming()
frame = np.zeros((480, 640, 3), dtype=np.uint8)
try:
    while True:
        frame[:] = 40
        cv2.putText(frame, time.strftime("%H:%M:%S"), (120, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        s.update_frame(frame)
        time.sleep(1 / 15)
finally:
    s.stop_streaming()
```

1. **Idle**: no FFmpeg starts, no "Streaming started" log — zero upstream.
2. **Wake**: open the viewer URL — ticking clock appears within ~10 s;
   console logs `Viewer demand detected - starting stream`.
3. **Teardown**: close the tab; ~30–35 s later the console logs
   `No viewer demand - stopping stream` and FFmpeg exits.

**Hook execution check** (should appear when a viewer connects):

```bash
sudo docker compose logs mediamtx | grep -i "on demand"
```

If you see "on demand command started" but the flag file never appears,
you are on the shell-less scratch image — use `latest-ffmpeg`.

---

## Tuning

- `runOnDemandStartTimeout` (40 s) must exceed the edge's worst-case cold
  start: `demand_poll_interval` + FFmpeg init (2 s) + RTSP connect
  (~1–2 s) + first keyframe (~1 s). Raise it if you raise the poll
  interval — tune the two together.
- Teardown latency ≈ `runOnDemandCloseAfter` + `demand_poll_interval` +
  `demand_grace_period`. Lower `runOnDemandCloseAfter` or `grace_period`
  to trim, but keep `grace_period` ≥ 5 s.

## Notes

- `mediamtx.yml` contains the Control API credentials (`transformsai`
  user). This repo is internal, but consider moving the password to an env
  var (`MTX_AUTHINTERNALUSERS_1_PASS`) if the repo's audience widens.
- The demand endpoint is public read-only. The flags leak nothing beyond
  "someone is watching camera X"; restrict by edge egress IP or a
  shared-secret header in `demand-nginx.conf` if that matters.
- No public reverse proxy at all? Change the demand port mapping to
  `"8880:80"` in `docker-compose.yml` and set
  `demand_url: "http://<host>:8880/demand/cam_sn_{camera_sn_id}"` on the
  edge instead.
