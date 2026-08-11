import json
import threading
from urllib.request import urlopen

import numpy as np

from applications.reconstruction.viz.live import LiveSimulation, serve


def test_live_canvas_registers_and_publishes():
    server = serve(port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_port}"
        simulation = LiveSimulation("test live", base_url)
        canvas = simulation.canvas("points", extent=2.0, point_radius=0.1)
        canvas.update(np.array([[0.0, 0.0], [0.5, -0.5]]), step=4, parameters={"level": [2, 3]})

        stream = urlopen(base_url + f"/api/simulations/{simulation.id}/contexts/points/events")
        snapshot = json.loads(stream.readline().removeprefix(b"data: "))
        assert snapshot["entries"][0]["coordinates"] == {"step": 4, "level": [2, 3]}
        stream.readline()  # ligne vide séparant les événements SSE
        canvas.update([[0.0, 0.0]], step=5, parameters={"level": [2, 3]})
        update = json.loads(stream.readline().removeprefix(b"data: "))
        assert update["entry"] == {
            "positions": [[0.0, 0.0]], "coordinates": {"step": 5, "level": [2, 3]},
        }
        stream.close()

        with urlopen(base_url + "/api/simulations") as response:
            simulations = json.load(response)
        assert simulations == [{
            "id": simulation.id,
            "name": "test live",
            "created_at": simulation.created_at,
            "contexts": ["points"],
        }]

        with urlopen(canvas.url) as response:
            assert b"EventSource" in response.read()
        with urlopen(base_url + "/latest") as response:
            assert response.url == canvas.url

        disks = simulation.canvas("disks", extent=2.0, geometry="disks")
        disks.update([[0.0, 0.0]], radii=0.2)
    finally:
        server.shutdown()
        server.server_close()
