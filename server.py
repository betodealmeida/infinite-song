"""
A server that generates an infinite stream of music.

https://stackoverflow.com/a/61506979/807118
"""

# pylint: disable=invalid-name

import asyncio
import random
import time
from datetime import datetime
from hashlib import md5
from io import BytesIO
from typing import Iterator

import numpy as np
import uvicorn
from fastapi import FastAPI
from pedalboard import Chorus, Delay, LadderFilter, Pedalboard, Reverb
from pydub import AudioSegment
from scipy import signal
from scipy.io import wavfile
from starlette.requests import Request
from starlette.responses import StreamingResponse

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

app = FastAPI()

FRAMERATE = 24000
PADS_VOLUME = 0.25
BASS_VOLUME = 0.4
NOTES_VOLUME = 0.8
PADS_DURATION = 16


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


def get_chord(timestamp: int, duration: int = PADS_DURATION) -> [str]:
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
    envelope = np.ones(FRAMERATE * PADS_DURATION) * sustain

    i = int(attack * FRAMERATE)
    envelope[:i] = np.linspace(0, 1, i)

    j = int(decay * FRAMERATE)
    envelope[i : i + j] = np.linspace(1, sustain, j)

    k = int(release * FRAMERATE)
    envelope[-k:] = np.linspace(sustain, 0, k)

    now = datetime.fromtimestamp(timestamp)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    t0 = int(start.timestamp())
    offset = ((timestamp - t0) % PADS_DURATION) * FRAMERATE

    return envelope[offset : offset + window]


def get_audio(
    timestamp: int,
    duration: int = 1,
) -> np.array:  # pylint: disable=too-many-locals
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
        k = 2 * np.pi / 17
        slow_vibrato = 0.6 + 0.2 * ((np.sin(timestamp * k) + 1) / 2)
        pads_buffer += audio * PADS_VOLUME * slow_vibrato

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

    return buffer.astype(np.float32)


async def response(duration: int = 1) -> Iterator[bytes]:
    """
    Encode the song on-the-fly as an MP3.
    """
    # align with a duration window
    now = int(time.time())
    timestamp = now - (now % duration) + duration
    await asyncio.sleep(timestamp - time.time())

    while True:
        buffer = BytesIO()
        wavfile.write(buffer, FRAMERATE, get_audio(timestamp, duration))
        buffer.seek(0)
        segment = AudioSegment.from_file(buffer, format="wav")

        ogg_chunk = BytesIO()
        segment.export(ogg_chunk, format="ogg")
        ogg_chunk.seek(0)
        yield ogg_chunk.getvalue()

        timestamp += duration
        now = int(time.time())
        offset = timestamp - now
        buffer = 30
        if offset > buffer:
            await asyncio.sleep(offset - buffer)


def generate_song(song_duration: int, filename: str) -> None:
    """
    Generate a song of a specific length.
    """
    window_duration = 1
    now = int(time.time())
    timestamp = now - (now % window_duration) + window_duration

    with open(filename, "wb") as fp:
        for _ in range(song_duration):
            chunk = get_audio(timestamp, window_duration)
            fp.write(chunk)
            timestamp += window_duration


@app.get("/stream.ogg")
async def stream() -> StreamingResponse:
    """
    Return an infinite stream of music.
    """
    return StreamingResponse(
        response(),
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "Transfer-Encoding": "chunked",
            "Accept-Ranges": "bytes",
        },
        media_type="audio/ogg.ogv",
    )


if __name__ == "__main__":
    uvicorn.run("server:app", port=8000, log_level="info", reload=True)
