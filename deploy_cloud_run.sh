#!/usr/bin/env bash
# Deploys the application to Google Cloud Run (container built by Cloud Build from the Dockerfile).
# Usage:  PROJECT_ID=my-project REGION=europe-west1 ./deploy_cloud_run.sh
# Prerequisite: gcloud CLI installed and authenticated (gcloud auth login).
set -euo pipefail

: "${PROJECT_ID:?Set PROJECT_ID}"
REGION="${REGION:-europe-west1}"
SERVICE="${SERVICE:-smart-student-performance}"
MAX_INSTANCES="${MAX_INSTANCES:-5}"
MIN_INSTANCES="${MIN_INSTANCES:-0}"
CONCURRENCY="${CONCURRENCY:-10}"     # simultaneous sessions per instance before a new instance is started

gcloud config set project "$PROJECT_ID"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com firestore.googleapis.com artifactregistry.googleapis.com

# One-time: create the Firestore database used for the prediction history (ignored if it already exists).
gcloud firestore databases create --location="$REGION" --type=firestore-native 2>/dev/null || true

gcloud run deploy "$SERVICE" \
  --source . \
  --region "$REGION" \
  --allow-unauthenticated \
  --port 8501 \
  --memory 1Gi --cpu 1 \
  --concurrency "$CONCURRENCY" \
  --min-instances "$MIN_INSTANCES" --max-instances "$MAX_INSTANCES" \
  --session-affinity \
  --timeout 3600 \
  --set-env-vars "HISTORY_BACKEND=firestore"

gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)'
