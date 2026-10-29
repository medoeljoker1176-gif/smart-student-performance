"""
Locust scenario for the scaling experiment (open the Locust web UI and ramp users up).

Run:  pip install locust
      locust -f loadtest/locustfile.py --host https://<service-url>
Streamlit pages are served over a WebSocket after the first HTTP request, so this scenario loads the
HTTP entry points (page shell, static assets and health check). It measures the cost of new sessions
arriving, which is what triggers Cloud Run to add instances; it does not click through the UI.
"""
from locust import HttpUser, between, task


class Visitor(HttpUser):
    wait_time = between(1, 3)

    @task(5)
    def open_app(self):
        self.client.get("/", name="home page")

    @task(1)
    def health(self):
        self.client.get("/_stcore/health", name="health check")
