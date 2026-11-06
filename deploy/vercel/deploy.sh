#!/usr/bin/env bash
# Deploy the web UI (read-only over the committed snapshot) to Vercel's free Hobby plan.
#
#   VERCEL_TOKEN=... deploy/vercel/deploy.sh          # token: vercel.com/account/tokens
#   FLI_VERCEL_BUILD_ONLY=1 deploy/vercel/deploy.sh   # only assemble the bundle (prints its path)
#
# The bundle is assembled in a temp directory so that only what the web process needs is uploaded:
# deploy/vercel/{app.py,requirements.txt,vercel.json}, src/fli, config/, the snapshot and the scoring
# model metadata. .env is never copied; FLI_BASIC_AUTH (from .env) is passed as a runtime variable and
# FLI_AUTH_ALL=1 puts every page behind the login, as on AWS.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
HERE="$REPO_DIR/deploy/vercel"
VERCEL_CLI="vercel@65.2.0"
NAME="${FLI_VERCEL_PROJECT:-frontier-lab-intel}"
log() { printf '[vercel] %s\n' "$*"; }

STAGE="$(mktemp -d)/$NAME"  # the directory name becomes the Vercel project name on first deploy
mkdir -p "$STAGE/data/snapshot" "$STAGE/data/models" "$STAGE/src"
cp "$HERE/app.py" "$HERE/requirements.txt" "$HERE/vercel.json" "$STAGE/"
echo "3.12" > "$STAGE/.python-version"
rsync -a --exclude '__pycache__' --exclude '*.pyc' "$REPO_DIR/src/fli" "$STAGE/src/"
rsync -a "$REPO_DIR/config" "$STAGE/"
cp "$REPO_DIR/data/snapshot/fli_snapshot.sqlite.gz" "$STAGE/data/snapshot/"
cp "$REPO_DIR"/data/models/scoring_*.json "$STAGE/data/models/"
[ -d "$HERE/.vercel" ] && cp -R "$HERE/.vercel" "$STAGE/"  # keep deploying to the same project
log "bundle at $STAGE ($(du -sh "$STAGE" | cut -f1))"
if [ "${FLI_VERCEL_BUILD_ONLY:-0}" = "1" ]; then echo "$STAGE"; exit 0; fi

: "${VERCEL_TOKEN:?set VERCEL_TOKEN (create one at https://vercel.com/account/tokens)}"
AUTH="$(grep -m1 '^FLI_BASIC_AUTH=' "$REPO_DIR/.env" 2>/dev/null | cut -d= -f2- | tr -d '"' || true)"
case "$AUTH" in
  *:????????????*) ;;
  *) echo "FLI_BASIC_AUTH=user:password (password >= 12 chars) must be set in .env"; exit 1 ;;
esac

cd "$STAGE"
OUT="$(npx --yes "$VERCEL_CLI" deploy --prod --yes --token "$VERCEL_TOKEN" \
  --env FLI_AUTH_ALL=1 --env "FLI_BASIC_AUTH=$AUTH" 2>&1)"
URL="$(grep -o 'https://[a-z0-9.-]*\.vercel\.app' <<<"$OUT" | tail -1)"  # the production alias is printed last
mkdir -p "$HERE/.vercel" && cp "$STAGE/.vercel/project.json" "$HERE/.vercel/"
log "deployed: $URL"
code="$(curl -s -o /dev/null -w '%{http_code}' "$URL/healthz" || true)"
log "health check /healthz -> $code"
