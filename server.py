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
from pedalboard import Chorus, Delay, LadderFilter, Pedalboard, Reverb

notes_fx = Pedalboard(
    [
        Delay(delay_seconds=1 / 3.0, feedback=0.4, mix=0.25),
        LadderFilter(mode=LadderFilter.Mode.LPF12, cutoff_hz=600),
        Reverb(room_size=0.5),
    ]
)
pads_fx = Pedalboard(
    [
        LadderFilter(mode=LadderFilter.Mode.LPF12, cutoff_hz=600),
        Reverb(room_size=1.0, wet_level=0.75, dry_level=0.25, width=1.0),
        Chorus(rate_hz=1 / 5.0),
    ]
)
bass_fx = Pedalboard(
    [
        LadderFilter(mode=LadderFilter.Mode.LPF12, cutoff_hz=300),
        Reverb(room_size=0.25),
    ]
)

app = Flask(__name__)

FRAMERATE = 24000
PAD_VOLUME = 0.25
BASS_VOLUME = 0.4
NOTES_VOLUME = 0.8
PAD_DURATION = 32


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


def get_envelope(  # pylint: disable=too-many-arguments
    timestamp: int,
    window: int,
    attack: float = 8.0,
    decay: float = 2.0,
    sustain: float = 0.8,
    release: float = 4.0,
) -> np.array:
    """
    Build an envelope for pads.

    TODO: allow passing ASDR parameters.
    """
    envelope = np.ones(FRAMERATE * PAD_DURATION) * sustain

    i = int(attack * FRAMERATE)
    envelope[:i] = np.linspace(0, 1, i)

    j = int(decay * FRAMERATE)
    envelope[i : i + j] = np.linspace(1, sustain, j)

    k = int(release * FRAMERATE)
    envelope[-k:] = np.linspace(sustain, 0, k)

    now = datetime.fromtimestamp(timestamp)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    t0 = int(start.timestamp())
    offset = ((timestamp - t0) % PAD_DURATION) * FRAMERATE

    return envelope[offset : offset + window]


def get_audio(
    timestamp: int, duration: int = 1
) -> bytes:  # pylint: disable=too-many-locals
    """
    Build music snippet.
    """
    hash_ = md5(str(timestamp).encode()).hexdigest()
    buffer = np.zeros(FRAMERATE * duration)

    # pads
    chord = get_chord(timestamp)
    pads_buffer = np.zeros(FRAMERATE * duration)
    for note in chord:
        frequency = get_frequency(note)

        t = np.linspace(timestamp, timestamp + duration, FRAMERATE * duration)
        audio = np.sin(2 * np.pi * frequency * t)
        audio *= get_envelope(timestamp - duration, FRAMERATE * duration)
        pads_buffer += audio * PAD_VOLUME

    buffer += pads_fx(pads_buffer, FRAMERATE, reset=False)

    # bass
    root = chord[0]
    frequency = get_frequency(root) / 4.0
    t = np.linspace(timestamp, timestamp + duration, FRAMERATE * duration)
    audio = signal.sawtooth(2 * np.pi * frequency * t)
    audio *= get_envelope(
        timestamp - duration,
        FRAMERATE * duration,
        attack=0.1,
        decay=0.5,
        sustain=0.9,
        release=0.5,
    )
    buffer += bass_fx(audio * BASS_VOLUME, FRAMERATE, reset=False)

    # generate 0-4 notes
    k = 2 * np.pi / 60
    drop_note = 4 + 8 * ((np.sin(timestamp * k) + 1) / 2)
    notes_buffer = np.zeros(FRAMERATE * duration)
    for i, c in enumerate(hash_[:4]):
        if int(c, 16) > drop_note:
            continue

        window = FRAMERATE // 4

        # 5 minute LFO controls duty of the notes
        k = 2 * np.pi / 301
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
            "F5",
            "G5",
        ]
        frequency = get_frequency(notes[int(hash_[i + 4], 16)])

        t = np.linspace(0, 1, window)
        audio = signal.sawtooth(2 * np.pi * frequency * t)
        audio[: int(window * (1 - duty))] = 0.0
        notes_buffer[i * window : (i + 1) * window] += audio * NOTES_VOLUME

    buffer += notes_fx(notes_buffer, FRAMERATE, reset=False)

    return buffer.astype(np.float32).tobytes()


def response(duration: int = 1) -> Iterator[bytes]:
    """
    Encode the song on-the-fly as an MP3.
    """
    with subprocess.Popen(
        f"ffmpeg -f f32le -acodec pcm_f32le -ar {FRAMERATE} -ac 1 -i pipe: -f mp3 pipe:".split(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as pipe:
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


def generate_song(duration: int, filename: str) -> None:
    """
    Generate a song of a specific length.
    """
    window = 16
    with open(filename, "wb") as fp:
        for i, chunk in enumerate(response(duration=window)):
            fp.write(chunk)
            if (i + 1) * window > duration:
                break


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
    # generate_song(29 * 24 * 60 * 60, "29_hour_long_song.mp3")
    app.run(host="0.0.0.0", port=8000, debug=True)
