"""
A server that generates an infinite stream of music.
"""

# pylint: disable=invalid-name

import random
import select
import subprocess
import time
from datetime import datetime
from hashlib import md5
from typing import Iterator

import numpy as np
from flask import Flask
from flask import Response
from scipy import signal
from pedalboard import Delay, LadderFilter, Pedalboard, Reverb

board = Pedalboard(
    [
        Delay(delay_seconds=1 / 3.0, feedback=0.4, mix=0.25),
        LadderFilter(mode=LadderFilter.Mode.LPF12, cutoff_hz=600),
        Reverb(room_size=0.5),
    ]
)

app = Flask(__name__)

FRAMERATE = 24000
PAD_VOLUME = 0.3
BASS_VOLUME = 0.4
PAD_DURATION = 16


def get_frequency(note: str, A4: int = 440) -> float:
    """
    Return the frequency of a note.

        >>> get_frequency("A4")
        440

    """
    notes = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]

    octave = int(note[-1])
    key_number = notes.index(note[:-1])

    if key_number < 3:
        key_number = key_number + 12 + ((octave - 1) * 12) + 1
    else:
        key_number = key_number + ((octave - 1) * 12) + 1

    return A4 * 2 ** ((key_number - 49) / 12)


def get_chord(timestamp: int, duration: int = PAD_DURATION) -> [str]:
    """
    Get the chord for a given instant.
    """
    # https://www.researchgate.net/figure/The-probabilities-for-the-second-chord-depending-on-the-first-chord_tbl1_324454816
    chords = [
        ["C4", "G4", "E5"],
        ["D4", "A4", "F5"],
        ["E4", "B4", "G5"],
        ["F4", "C4", "A5"],
        ["G4", "D4", "B5"],
        ["A4", "E4", "C5"],
    ]
    chord_probability = np.array(
        [
            [24, 35, 0, 20, 70, 5],
            [2, 2, 5, 1, 1, 5],
            [2, 1, 0, 1, 2, 1],
            [39, 4, 85, 1, 13, 49],
            [20, 86, 2, 76, 1, 39],
            [35, 4, 8, 1, 14, 1],
        ]
    ).cumsum(axis=0)
    chord_probability = chord_probability / chord_probability.max(axis=0)

    now = datetime.fromtimestamp(timestamp)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    t0 = int(start.timestamp())
    random.seed(t0)

    chord = random.randint(0, 5)
    while t0 + duration < timestamp:
        choice = random.random()
        for i, probability in enumerate(chord_probability[:, chord]):
            if choice < probability:
                chord = i
                break
        t0 += duration

    return chords[chord]


def get_envelope(timestamp: int, window: int) -> np.array:
    """
    Build an envelope for pads.

    TODO: allow passing ASDR parameters.
    """
    attack = 2
    decay = 1
    sustain = 0.8
    release = 1

    envelope = np.ones(FRAMERATE * PAD_DURATION) * sustain

    i = attack * FRAMERATE
    envelope[:i] = np.linspace(0, 1, i)

    j = decay * FRAMERATE
    envelope[i : i + j] = np.linspace(1, sustain, j)

    k = release * FRAMERATE
    envelope[-k:] = np.linspace(sustain, 0, k)

    offset = timestamp % PAD_DURATION

    return envelope[offset : offset + window]


def get_audio(timestamp: int, duration: int = 1) -> bytes:
    """
    Build music snippet.
    """
    hash_ = md5(str(timestamp).encode()).hexdigest()
    buffer = np.zeros(FRAMERATE * duration)

    # pads
    chord = get_chord(timestamp)
    for note in chord:
        frequency = get_frequency(note)

        t = np.linspace(timestamp, timestamp + duration, FRAMERATE * duration)
        audio = np.sin(2 * np.pi * frequency * t)
        audio *= get_envelope(timestamp, FRAMERATE * duration)
        buffer += audio * PAD_VOLUME

    # bass
    root = chord[0]
    frequency = get_frequency(root) / 4.0
    t = np.linspace(timestamp, timestamp + duration, FRAMERATE * duration)
    audio = signal.sawtooth(2 * np.pi * frequency * t)
    buffer += audio * BASS_VOLUME

    # generate 0-4 notes
    for i, c in enumerate(hash_[:4]):
        if int(c, 16) % 2 == 0:
            continue

        window = FRAMERATE // 4

        # 5 minute LFO controls duty of the notes
        k = 2 * np.pi / (5 * 60)
        duty = ((np.sin(timestamp * k) * 0.8) + 1) / 2

        # choose note frequency
        notes = [
            "A3",
            "C3",
            "D3",
            "E3",
            "G3",
            "A4",
            "C4",
            "D4",
            "E4",
            "G4",
            "A5",
            "C5",
            "D5",
            "E5",
            "G5",
            "A5",
        ]
        frequency = get_frequency(notes[int(hash_[i + 4], 16)])

        t = np.linspace(0, 1, window)
        audio = signal.sawtooth(2 * np.pi * frequency * t)
        audio[: int(window * (1 - duty))] = 0.0
        buffer[i * window : (i + 1) * window] += audio

    processed = board(buffer, FRAMERATE, reset=False)

    return processed.astype(np.float32).tobytes()


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
        duration = 1

        # align with a duration window
        now = int(time.time())
        timestamp = now - (now % duration) + duration
        time.sleep(timestamp - time.time())

        poll = select.poll()
        poll.register(pipe.stdout, select.POLLIN)

        while True:
            pipe.stdin.write(get_audio(timestamp, duration))
            timestamp += duration
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
