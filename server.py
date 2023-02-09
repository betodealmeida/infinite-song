"""
A server that generates an infinite stream of music.
"""

import select
import subprocess
import time
from typing import Iterator

import numpy as np
from flask import Flask
from flask import Response
from scipy import signal
from pedalboard import LadderFilter, Pedalboard, Reverb

board = Pedalboard(
    [
        LadderFilter(mode=LadderFilter.Mode.LPF12, cutoff_hz=900),
        Reverb(room_size=0.25),
    ]
)

app = Flask(__name__)

FRAMERATE = 24000


def get_audio() -> bytes:
    """
    Build music snippet.
    """
    t = np.linspace(0, 1, FRAMERATE)
    freq = 220
    audio = signal.sawtooth(2 * np.pi * freq * t)
    audio[FRAMERATE // 4 :] = 0.0
    audio = board(audio, FRAMERATE, reset=False)
    buf = audio.astype(np.float32)

    return buf.tobytes()


def response() -> Iterator[bytes]:
    """
    Encode the song on-the-fly as an MP3.
    """
    with subprocess.Popen(
        f"ffmpeg -f f32le -acodec pcm_f32le -ar {FRAMERATE} -ac 1 -i pipe: -f mp3 pipe:".split(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as pipe:
        poll = select.poll()
        poll.register(pipe.stdout, select.POLLIN)
        while True:
            pipe.stdin.write(get_audio())
            while poll.poll(0):
                yield pipe.stdout.readline()


@app.route("/stream.mp3", methods=["GET"])
def stream() -> Response:
    """
    Return an infinite stream of music.
    """
    return Response(
        response(),
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
        mimetype="audio/mpeg",
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
