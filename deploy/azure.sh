#!/usr/bin/env bash
# Part 6: deploy the Docker image to Azure Container Apps (scales to zero when idle).
#
#   deploy/azure.sh push     build the amd64 image and push it to ghcr.io (no Azure needed)
#   deploy/azure.sh deploy   create or update the Azure app; prints the public URL
#
# Needs: docker, gh (token with write:packages), az (logged in with `az login`).
# While the image is private, set GHCR_READ_TOKEN to a GitHub token with only read:packages
# (Azure stores it; the gh login token can see every private repo, so it isn't used here).
# Azure's servers are amd64; the M2 is arm64, so the image is built for linux/amd64.
set -euo pipefail

APP=banking-intent-router
IMAGE=ghcr.io/dombayj/$APP
TAG=$(git rev-parse --short HEAD)
GROUP=$APP-rg
ENV=$APP-env
LOCATION=${LOCATION:-westeurope}  # Azure for Students may only allow some regions

push() {
  gh auth token | docker login ghcr.io -u dombayj --password-stdin
  docker buildx build --platform linux/amd64 -t "$IMAGE:$TAG" -t "$IMAGE:latest" --push .
}

deploy() {
  az extension add --name containerapp --upgrade --only-show-errors
  az provider register --namespace Microsoft.App --wait
  az group create --name "$GROUP" --location "$LOCATION" --output none
  # No Log Analytics workspace: it bills per GB of logs, and the app logs nothing useful.
  az containerapp env show --name "$ENV" --resource-group "$GROUP" --output none 2>/dev/null ||
    az containerapp env create --name "$ENV" --resource-group "$GROUP" \
      --location "$LOCATION" --logs-destination none --output none
  if az containerapp show --name "$APP" --resource-group "$GROUP" --output none 2>/dev/null; then
    az containerapp update --name "$APP" --resource-group "$GROUP" \
      --image "$IMAGE:$TAG" --output none
  else
    # min-replicas 0 = scale to zero when idle (free), 1 vCPU / 2 GiB for a faster cold start.
    az containerapp create --name "$APP" --resource-group "$GROUP" --environment "$ENV" \
      --image "$IMAGE:$TAG" \
      ${GHCR_READ_TOKEN:+--registry-server ghcr.io --registry-username dombayj --registry-password "$GHCR_READ_TOKEN"} \
      --target-port 8000 --ingress external \
      --cpu 1 --memory 2Gi --min-replicas 0 --max-replicas 1 --output none
  fi
  echo "https://$(az containerapp show --name "$APP" --resource-group "$GROUP" \
    --query properties.configuration.ingress.fqdn --output tsv)"
}

"${1:?usage: deploy/azure.sh push|deploy}"
