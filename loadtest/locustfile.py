from locust import HttpUser, task, between


class Traffic(HttpUser):
    wait_time = between(0.05, 0.2)

    @task
    def hit_router(self):
        self.client.get("/route")
