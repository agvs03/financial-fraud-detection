# Running this repo's Vela pipeline

The [`.vela.yml`](../.vela.yml) at the repo root defines a [Vela](https://go-vela.github.io/docs/)
CI/CD pipeline (install → pipeline smoke test → tests), mirroring the GitHub
Actions workflows.

**Important:** Vela is **self-hosted** — there is no hosted Vela cloud you can
just switch on for a GitHub repo (the way GitHub Actions works). To execute this
pipeline you run the Vela control plane yourself and enroll this repo. The
`.vela.yml` file does nothing on its own.

## What you need to run it

1. **A GitHub OAuth App** — Vela authenticates and receives webhooks through it.
   Create one at GitHub → Settings → Developer settings → OAuth Apps. Note the
   client ID/secret and set the callback URL to your Vela server's
   `/authenticate/callback`.

2. **A running Vela server + worker.** The official quickstart uses Docker
   Compose (Vela server, a worker, a database, and Redis/Valkey for the queue):
   <https://go-vela.github.io/docs/installation/> → *"Quickstart"*. Because
   GitHub must reach your server with webhooks, expose it publicly or use a
   tunnel (e.g. `cloudflared` / `ngrok`) while testing locally.

3. **Enroll the repo.** Log into the Vela UI, enable
   `agvs03/financial-fraud-detection`. Vela installs a webhook so each push
   triggers the pipeline.

4. **Push** (or click *Restart* in the UI) and watch the three steps run in
   Vela's build view.

## Notes for this pipeline

- Each Vela step runs in its own `python:3.11` container sharing `/vela/src`.
  Python packages don't persist across steps, so the install step creates
  `.venv` **in the workspace** and later steps re-activate it.
- The smoke test installs TensorFlow and trains on synthetic data — it takes a
  few minutes and needs a worker with enough memory/disk (same as the GitHub
  Actions "Pipeline" job).

## Why both Vela and GitHub Actions?

GitHub Actions already runs this repo's CI end to end and needs no
infrastructure. This Vela pipeline is here to demonstrate the same checks
expressed in a second, self-hosted CI/CD system — it is **optional** and adds
no functionality GitHub Actions doesn't already provide. Keep it if you want to
run/showcase Vela; otherwise the repo is fully covered without it.
