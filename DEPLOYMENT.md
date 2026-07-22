# Deployment Guide

How the Kaya web app is served to other computers, and how the CI/CD pipeline
deploys it. The box is **serving-only**: `kaya-prod` is the **always-on**
production app (auto-recovers after reboot). "Push to prod" rebuilds and restarts
the live container from a dedicated `~/kaya-prod` checkout. `kaya-dev` exists for
occasional manual testing and shares the single GPU, so only one runs at a time.

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
kaya-prod / kaya-dev container (Gradio + WhatsApp webhook)
   │  Gradio username/password                              [protection layer 2]
   ▼
RAG (GPU) ──► generation backend
                ├─ hf:   in-process Unsloth model (dev default)
                └─ gguf: llama.cpp `llama` container over HTTP (prod, ~15× faster)
```

- **dev**: `dev.kaya.example.com` → `kaya-dev:7861`
- **prod**: `kaya.example.com` → `kaya-prod:7860`
- dev and prod **share one GPU** — only run one at a time (`app_up.sh` enforces this).
- **Inference backend:** prod runs the `gguf` backend (`KAYA_INFERENCE_BACKEND=gguf` on `kaya-prod`), so generation happens in the `llama` compose service (`gguf` profile) serving `models/gguf/kaya-wpp-Q6_K.gguf`. `deploy_prod.sh` starts it automatically. Roll back with `KAYA_INFERENCE_BACKEND=hf scripts/deploy_prod.sh`. Dev defaults to `hf` (in-process) — no `llama` service needed.

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

## Prod runs from its own checkout (`~/kaya-prod`)

Prod is **always-on** and serves from a dedicated checkout, separate from where you
develop (`~/Desktop/KayaChatBot`). This is what makes "push to prod" actually update
the live site, and lets you keep editing without affecting it.

**One-time setup on the box:**
```bash
git clone git@github.com:GustavoPintoDeAbreu/KayaChatBot.git ~/kaya-prod
ln -s ~/Desktop/KayaChatBot/models ~/kaya-prod/models   # share the 42GB models (symlink, no copy)
ln -s ~/Desktop/KayaChatBot/data   ~/kaya-prod/data     # share data/rag_db
cp  ~/Desktop/KayaChatBot/.env     ~/kaya-prod/.env     # or let CI write it from prod secrets
sudo systemctl enable docker                            # so prod auto-starts after a reboot
```

## The pipeline

| Stage | Trigger | Workflow | What it does |
|---|---|---|---|
| Test | PR to `main` | `ci.yml` | Builds the image, runs `pytest` in `kaya-test`. Merge gate. |
| Validate | merge to `main` | `validate-main.yml` | Rebuilds + runs the test suite so main is known-deployable. No container start. |
| Deploy prod | manual (`workflow_dispatch`) | `deploy-prod.yml` | Pauses for `prod` approval, writes `.env` from `prod` secrets into `~/kaya-prod`, then runs `scripts/deploy_prod.sh` → **rebuilds and restarts the live prod container** on the chosen ref. |

Flow: open PR → CI + review → merge → `main` validated automatically → run
**Deploy (prod)** → approve the gate → the live site is now on that commit.

**What's live:** the UI header shows the env + commit, and `scripts/app_status.sh`
shows the running container. To make a specific version live, pass its ref to
**Deploy (prod)** (defaults to `main`).

---

## Runbook

```bash
# Make a commit live (the normal "push to prod"); CI's Deploy (prod) calls this.
scripts/deploy_prod.sh             # deploy main
scripts/deploy_prod.sh <ref>       # deploy a specific commit/tag/branch

# Manually power an env up/down (also starts/stops the Cloudflare Tunnel).
scripts/app_up.sh prod             # → http://localhost:7860 + prod hostname
scripts/app_up.sh dev              # → http://localhost:7861 + dev hostname (stops prod first; one GPU)
scripts/app_down.sh prod|dev|all   # stop and free the GPU
scripts/app_status.sh              # running containers + GPU usage

# Follow logs (model load takes ~1 min)
docker compose logs -f kaya-prod
```

**Reboot recovery:** `kaya-prod` and `cloudflared` use `restart: unless-stopped`,
so with `sudo systemctl enable docker` the site comes back automatically after a
reboot/power-cycle — no manual step. (If the box is *off*, the site is down and
visitors get a Cloudflare tunnel error until it's back.)

**One GPU rule:** the box is serving-only and the GPU fits one model at a time.
Running the `dev` container or fine-tuning means stopping prod first
(`deploy_prod.sh`/`app_up.sh` stop the other env automatically). For quick local
iteration, prefer the venv (`kaya_chatbot_env/bin/python ...`) and `pytest`, which
don't need the GPU and don't disturb the live site.

### Rotating the Gradio password / tunnel token
Update the value in the `prod` (and `dev`) GitHub environment secrets **and** in
`~/kaya-prod/.env`, then redeploy: `scripts/deploy_prod.sh`.

---

## Local quick test (no Cloudflare)

```bash
cp .env.example .env   # fill in KAYA_WEB_USER/PASS at minimum
scripts/app_up.sh dev
# open http://<box-LAN-IP>:7861 from another computer on the same network
```

---

## Troubleshooting (gotchas hit during setup)

- **`cloudflared`: "Provided Tunnel token is not valid."** You copied the
  **truncated** token. The dashboard shows it abbreviated (e.g. `eyJhIjoi...In0=`);
  hand-selecting that copies the literal `...`. Use the **Copy button on the full
  connector install command** and take the whole `--token eyJ…` value. Also strip
  any trailing punctuation — a stray `.` makes it invalid. Tokens are base64
  (`eyJ…`), ~250 chars, and never contain `...` or end in `.`.

- **Cloudflare Access: "Unable to find your Access application / invalid URL."**
  The hostname→app binding went stale (common with the **"Published application
  routes" (Beta)** tunnel feature). Fix: **delete the Access application and
  recreate it fresh** — a new app gets a clean AUD and binds correctly. When
  creating it, choose destination type **Public DNS** for a public hostname
  (`dev.sigmakayachat.pt`), not "Private destinations".

- **Tunnel routes must use your real domain.** If a route still points at a
  placeholder like `example.com`, the hostname won't resolve. Each public
  hostname route is `<sub>.<your-domain>` → `http://kaya-dev:7861` (dev) /
  `http://kaya-prod:7860` (prod). The connector picks up route edits live — no
  restart needed.

- **502 at the public URL.** The target container isn't powered up — the app is
  on-demand. Run `scripts/app_up.sh dev` (or `prod`).

- **One GPU.** dev and prod can't both hold the model; `app_up.sh` refuses to
  start one while the other runs. Stop the other first (`scripts/app_down.sh`).

- **Verifying from the box without logging in:** `curl -sI https://dev.<domain>`
  — a 302 to `*.cloudflareaccess.com/.../login/...` means Access is gating it
  correctly; the "Unable to find application" body means the binding is stale
  (recreate the app, above).

- **Ungrounded answers / DMs ignored after a deploy.** `~/kaya-prod/data/` must
  hold the gitignored runtime files: `rag_db/`, `group_members.json`,
  `whatsapp_whitelist.json`, `whatsapp_contacts.json`. If `data/` is a **real dir**
  (git materialised the tracked `*.example.json`) instead of the intended
  `ln -s ~/Desktop/KayaChatBot/data ~/kaya-prod/data` symlink, those files are
  missing → RAG init fails (hallucinated answers) and the DM whitelist is empty
  (every direct message silently ignored). Fix: copy them from the dev `data/`
  (leave `data/waha/` — the linked-device session — untouched), then
  `docker restart kaya-prod`. Boot log should show `RAG Retriever initialized` and
  `Loaded N WhatsApp whitelist number(s)`.

- **gguf backend: bot never replies / errors on generate.** The `llama` service
  isn't up. `deploy_prod.sh` starts it, or manually
  `docker compose --profile gguf up -d llama`; check `docker logs kaya-llama` for
  `model loaded`. The app reaches it at `http://llama:8080` on the compose network.
  Requires `models/gguf/kaya-wpp-Q6_K.gguf` to exist (shared via the `models`
  symlink). To bypass entirely, redeploy with `KAYA_INFERENCE_BACKEND=hf`.
