# Cloud Deployment (Google Cloud Run + Cloud Firestore)

This guide deploys the application as a container on a managed, auto-scaling cloud service. It is the
cloud-computing version of the project: the machine-learning code is unchanged; what is added is the
deployment, the durable storage and the evaluation procedure.

## Architecture

```
 Browser ──HTTPS──▶ Cloud Run service (managed load balancer, TLS, autoscaling 0..N)
                          │  container image built by Cloud Build from the Dockerfile
                          │  (Streamlit app + pre-trained scikit-learn model baked into the image)
                          ▼
                    Cloud Firestore  (prediction history shared by all instances)
 Container images ─▶ Artifact Registry          Metrics and logs ─▶ Cloud Monitoring / Cloud Logging
```

| Component | Role | Cloud concept |
|---|---|---|
| Cloud Run | Runs the container, scales instances with the number of concurrent sessions | Serverless / PaaS, elasticity, pay-per-use |
| Cloud Build + Artifact Registry | Builds and stores the container image | Containerisation, CI/CD |
| Cloud Firestore | Stores the prediction history outside the instances | Managed NoSQL, durability, statelessness |
| Cloud Monitoring | Instance count, request latency, CPU and memory charts | Monitoring, observability |
| IAM | Controls who can deploy and which identity the service uses | Cloud security |

Why Firestore: a Cloud Run instance has an ephemeral, per-instance file system. With the original CSV file,
history was lost on restart and differed between instances. `HISTORY_BACKEND=firestore` makes the service
stateless, which is what allows it to scale out safely.

## Steps

1. Create a Google Cloud project, enable billing, and install and authenticate the `gcloud` CLI.
2. From the project folder run:
   ```bash
   PROJECT_ID=<your-project-id> REGION=europe-west1 ./deploy_cloud_run.sh
   ```
   The script enables the required APIs, creates the Firestore database, builds the image, deploys the
   service (1 vCPU, 1 GiB, 10 concurrent sessions per instance, 0 to 5 instances, session affinity because
   Streamlit uses WebSockets) and prints the public URL.
3. Open the URL, make a prediction on the **Prediction** page and open **History**. Check in the Firestore
   console that a document appeared in the `prediction_history` collection.
4. Optional: to require sign-in, add the `[auth]` block as environment variables or a Secret Manager secret
   (see `.streamlit/secrets.toml.example`).

Local run without any cloud service is unchanged: `streamlit run app.py` (history stays in a local CSV).

## Measuring the system (all numbers must come from your own runs)

Follow `docs/CLOUD_EVALUATION.md`. In short:

```bash
python loadtest/measure.py https://<service-url> --requests 200 --concurrency 20     # latency, throughput, errors
locust -f loadtest/locustfile.py --host https://<service-url>                        # ramp users up, watch instances
```
Watch **Cloud Run → Metrics** (container instance count, request latency, CPU) while the test runs and take screenshots.

## Screenshots to capture (10 to 12, each with a numbered caption)

1. Google Cloud project dashboard
2. Enabled APIs (Cloud Run, Cloud Build, Firestore)
3. Cloud Build log of the image build
4. Artifact Registry showing the image
5. Cloud Run service details (CPU, memory, concurrency, min and max instances)
6. The live application with the public URL visible
7. Prediction result page (at-risk student)
8. Firestore console showing history documents
9. Cloud Run metrics at low load (1 instance)
10. Cloud Run metrics during the load test (instance count increased)
11. Locust or `measure.py` output
12. Cloud Run logs or the billing / cost page

## Clean-up (avoid charges)

```bash
gcloud run services delete smart-student-performance --region europe-west1
```
Delete the Firestore collection and the project when the evaluation is finished.
