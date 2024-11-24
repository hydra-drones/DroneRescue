from flask import Flask, render_template_string, request, jsonify
import threading

app = Flask(__name__)

# Game State
player_position = {"x": 1, "y": 1}
flag_position = {"x": 8, "y": 8}
obstacles = [{"x": 3, "y": 3}, {"x": 5, "y": 5}, {"x": 7, "y": 2}]
vision_radius = 2  # Define how far the player can see

# HTML + JavaScript Template
template = """
<!DOCTYPE html>
<html>
<head>
    <title>Simple Dot Game</title>
    <style>
        #gameCanvas {
            border: 1px solid black;
        }
    </style>
    <script>
        document.addEventListener("DOMContentLoaded", function() {
            const canvas = document.getElementById("gameCanvas");
            const ctx = canvas.getContext("2d");
            const cellSize = 50;

            function drawGame(state) {
                ctx.clearRect(0, 0, canvas.width, canvas.height);

                // Draw visible obstacles
                ctx.fillStyle = "black";
                state.obstacles.forEach(obstacle => {
                    if (Math.abs(obstacle.x - state.player.x) <= state.vision_radius &&
                        Math.abs(obstacle.y - state.player.y) <= state.vision_radius) {
                        ctx.fillRect((obstacle.x - state.player.x + state.vision_radius) * cellSize,
                                     (obstacle.y - state.player.y + state.vision_radius) * cellSize, cellSize, cellSize);
                    }
                });

                // Draw flag if within vision radius
                if (Math.abs(state.flag.x - state.player.x) <= state.vision_radius &&
                    Math.abs(state.flag.y - state.player.y) <= state.vision_radius) {
                    ctx.fillStyle = "red";
                    ctx.fillRect((state.flag.x - state.player.x + state.vision_radius) * cellSize,
                                 (state.flag.y - state.player.y + state.vision_radius) * cellSize, cellSize, cellSize);
                }

                // Draw player in the center of the vision area
                ctx.fillStyle = "blue";
                ctx.fillRect(state.vision_radius * cellSize, state.vision_radius * cellSize, cellSize, cellSize);
            }

            function updateGame(direction) {
                fetch('/move', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ direction: direction })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.gameOver) {
                        alert("Game Over!");
                    } else if (data.won) {
                        alert("You Win!");
                    }
                    drawGame(data.state);
                });
            }

            document.addEventListener("keydown", function(event) {
                if (event.key === "ArrowUp") {
                    updateGame("up");
                } else if (event.key === "ArrowDown") {
                    updateGame("down");
                } else if (event.key === "ArrowLeft") {
                    updateGame("left");
                } else if (event.key === "ArrowRight") {
                    updateGame("right");
                }
            });

            // Initial draw
            drawGame({ player: { x: 1, y: 1 }, flag: { x: 8, y: 8 }, obstacles: [ { x: 3, y: 3 }, { x: 5, y: 5 }, { x: 7, y: 2 } ], vision_radius: 2 });
        });
    </script>
</head>
<body>
    <canvas id="gameCanvas" width="300" height="300"></canvas>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(template)


@app.route("/move", methods=["POST"])
def move():
    global player_position
    data = request.get_json()
    direction = data.get("direction")

    # Update player position based on direction
    new_position = player_position.copy()
    if direction == "up":
        new_position["y"] -= 1
    elif direction == "down":
        new_position["y"] += 1
    elif direction == "left":
        new_position["x"] -= 1
    elif direction == "right":
        new_position["x"] += 1

    # Check for collisions with obstacles
    if any(
        obstacle["x"] == new_position["x"] and obstacle["y"] == new_position["y"]
        for obstacle in obstacles
    ):
        return jsonify(
            {
                "gameOver": True,
                "state": {
                    "player": player_position,
                    "flag": flag_position,
                    "obstacles": obstacles,
                    "vision_radius": vision_radius,
                },
            }
        )

    # Check if player reached the flag
    if new_position == flag_position:
        return jsonify(
            {
                "won": True,
                "state": {
                    "player": player_position,
                    "flag": flag_position,
                    "obstacles": obstacles,
                    "vision_radius": vision_radius,
                },
            }
        )

    # Update player position
    player_position = new_position
    return jsonify(
        {
            "state": {
                "player": player_position,
                "flag": flag_position,
                "obstacles": obstacles,
                "vision_radius": vision_radius,
            }
        }
    )


if __name__ == "__main__":
    # Start the Flask app in a separate thread
    threading.Thread(target=lambda: app.run(debug=True, use_reloader=False)).start()
