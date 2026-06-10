# Deployment Guide

How the Kaya web app is served to other computers, and how the CI/CD pipeline
deploys it. The app runs **on demand** on a single RTX 3090 box — it is not
running 24/7. Deployment *stages* a release; you power it up when you want it.

---

## Architecture

```
Browser (any computer)
   │  https://kaya.example.com
   ▼
Cloudflare Access  ──►  login page (allowed emails only)   [protection layer 1]
   │
   ▼
Cloudflare Tunnel  ──►  cloudflared container (no inbound ports on the box)
   │  http://kaya-prod:7860  (compose network)
   ▼
kaya-prod / kaya-dev container (Gradio)
   │  Gradio username/password                              [protection layer 2]
   ▼
fine-tuned model + RAG  (GPU)
```

- **dev**: `dev.kaya.example.com` → `kaya-dev:7861`
- **prod**: `kaya.example.com` → `kaya-prod:7860`
- dev and prod **share one GPU** — only run one at a time (`app_up.sh` enforces this).

---

## One-time setup

### 1. Cloudflare (outside the repo)

1. Add your domain to Cloudflare (a free plan works).
2. **Zero Trust → Networks → Tunnels → Create a tunnel** (remotely-managed).
   Copy the **tunnel token** — this becomes the `CLOUDFLARE_TUNNEL_TOKEN` secret.
3. On the tunnel, add **Public Hostnames**:
   | Hostname | Service |
   |---|---|
   | `kaya.example.com` | `http://kaya-prod:7860` |
   | `dev.kaya.example.com` | `http://kaya-dev:7861` |
   The `cloudflared` container shares the compose network, so it resolves the
   `kaya-prod` / `kaya-dev` service names.
4. **Zero Trust → Access → Applications → Add a self-hosted application** for each
   hostname. Add a policy that **allows only specific emails** (Action: Allow,
   Include: Emails → your group's addresses). This is the Cloudflare login page.

### 2. Self-hosted GitHub Actions runner (on the GPU box)

1. Repo → **Settings → Actions → Runners → New self-hosted runner**. Follow the
   install steps; give it the labels `self-hosted` and `gpu`.
2. Install it as a service so it survives reboots:
   ```bash
   ./svc.sh install
   ./svc.sh start
   ```
3. Make sure the runner's user can use Docker + the NVIDIA runtime:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
   ```
4. The 42 GB of models live at `./models` on the box and are bind-mounted into
   the containers — **CI never transfers them**.

### 3. GitHub Environments, secrets, and branch protection

1. **Settings → Environments**: create `dev` and `prod`.
   - On `prod`, enable **Required reviewers** (add yourself). This pauses the prod
     deploy for manual approval.
2. Add these secrets to **both** environments (values differ per env as needed):
   | Secret | Used for |
   |---|---|
   | `XAI_API_KEY` | LLM provider |
   | `AZURE_OPENAI_API_KEY_gpt_41_mini` | LLM provider |
   | `AZURE_OPENAI_API_KEY_gpt_53_chat` | LLM provider |
   | `KAYA_WEB_USER` | Gradio login user |
   | `KAYA_WEB_PASS` | Gradio login password |
   | `CLOUDFLARE_TUNNEL_TOKEN` | Cloudflare Tunnel |
3. **Settings → Branches → Branch protection** on `main`: require a PR review and
   require the **CI** check to pass before merging.

---

## The pipeline

| Stage | Trigger | Workflow | What it does |
|---|---|---|---|
| Test | PR to `main` | `ci.yml` | Builds the image, runs `pytest` in the `kaya-test` container. Merge gate. |
| Deploy dev | merge to `main` | `deploy-dev.yml` | Writes `.env` from `dev` secrets, rebuilds, runs the test gate, **stages** the dev release. |
| Deploy prod | manual (`workflow_dispatch`) | `deploy-prod.yml` | Pauses for `prod` approval, then writes `.env` from `prod` secrets, rebuilds, tests, **stages** the prod release. |

Flow: open PR → CI + review → merge → dev staged automatically → test it (power
up dev) → run **Deploy (prod)** → approve the gate → power up prod.

Each deploy workflow has an optional `start: true` input to power the app up
automatically after staging; otherwise it stays off until you run `app_up.sh`.

---

## Power-up / power-down runbook

```bash
# Start (also starts the Cloudflare Tunnel). Refuses if the other env is up.
scripts/app_up.sh dev      # → http://localhost:7861  + dev.kaya.example.com
scripts/app_up.sh prod     # → http://localhost:7860  + kaya.example.com

# Check what's running and GPU usage
scripts/app_status.sh

# Stop and free the GPU
scripts/app_down.sh dev
scripts/app_down.sh prod
scripts/app_down.sh all    # stop both apps + the tunnel

# Follow logs (model load takes ~1 min)
docker compose logs -f kaya-prod
```

**One GPU rule:** dev and prod cannot both hold the model. `app_up.sh` refuses to
start one while the other is running — stop the other first.

### Rotating the Gradio password
Update `KAYA_WEB_USER` / `KAYA_WEB_PASS` in the GitHub environment secrets and the
box's `.env`, then `app_down.sh <env> && app_up.sh <env>`.

### Always-on dev (optional)
If you later want dev to stay up across reboots, the `kaya-dev`/`cloudflared`
services already set `restart: unless-stopped` on the tunnel; add the same to
`kaya-dev` and start it once with `docker compose --profile dev --profile tunnel up -d`.

---

## Local quick test (no Cloudflare)

```bash
cp .env.example .env   # fill in KAYA_WEB_USER/PASS at minimum
scripts/app_up.sh dev
# open http://<box-LAN-IP>:7861 from another computer on the same network
```
