import os
import unittest

os.environ["PUBLIC_DEMO_MODE"] = "true"

import app as app_module


class PublicDemoModeTests(unittest.TestCase):
    def setUp(self):
        self.client = app_module.app.test_client()

    def test_public_demo_disables_camera_controls(self):
        response = self.client.post("/api/camera/start")

        self.assertEqual(response.status_code, 403)
        self.assertFalse(response.get_json()["success"])

    def test_public_demo_disables_remote_camera_configuration(self):
        source = app_module.camera.source

        response = self.client.post(
            "/api/camera/config",
            json={"source": "rtsp://example.invalid/stream"}
        )

        self.assertEqual(response.status_code, 403)
        self.assertEqual(app_module.camera.source, source)

    def test_public_demo_disables_record_verification(self):
        response = self.client.post(
            "/api/records/1/verify",
            json={"action": "REJECT"}
        )

        self.assertEqual(response.status_code, 403)

    def test_image_view_does_not_serve_files_outside_allowed_directories(self):
        response = self.client.get("/image_view", query_string={"path": "app.py"})

        self.assertEqual(response.status_code, 404)

if __name__ == "__main__":
    unittest.main()

