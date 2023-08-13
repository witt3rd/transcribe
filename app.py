"""
This script provides a command-line interface for transcribing audio and video files using the whisper speech recognition model.

"""
import json
import locale
import os
from pathlib import Path
import re

#
from dotenv import load_dotenv
import ffmpeg
from pyannote.audio import Pipeline
from pydub import AudioSegment
import torch
import typer
import whisper
import yt_dlp


#
# Environment variables and general configuration
#
locale.getpreferredencoding = lambda: "UTF-8"

load_dotenv()  # take environment variables from .env.

HUGGINGFACE_ACCESS_TOKEN = os.getenv("HUGGINGFACE_ACCESS_TOKEN")
TRANSCRIPTS_PATH = str(Path(os.getenv("TRANSCRIPTS_PATH", default="transcripts")))
MEDIA_PATH = str(Path(os.getenv("MEDIA_PATH", default="videos")))

CHECK_EMOJII = "\u2705"
HOURGLASS_EMOJII = "\u23F3"

Path(TRANSCRIPTS_PATH).mkdir(parents=True, exist_ok=True)

#
#
#


def download_youtube(
    url: str,
) -> Path:
    """
    Download a YouTube video and save it to disk.

    Args:
        url (str): The URL of the YouTube video to download.

    Returns:
        Path: The path to the downloaded video file.

    Raises:
        ValueError: If the URL is invalid or the video cannot be downloaded.

    Example:
        >>> download_youtube('https://www.youtube.com/watch?v=dQw4w9WgXcQ')
        Path('/path/to/dQw4w9WgXcQ.mp4')
    """
    if "youtu.be" in url:
        video_id = url.split("/")[-1]
    else:
        video_id = url.split("v=")[-1]
        if "&" in video_id:
            video_id = video_id.split("&")[0]

    output_path = Path(MEDIA_PATH).joinpath(f"{video_id}.mp4")
    if output_path.exists():
        print(f"{CHECK_EMOJII} Aleady downloaded: {output_path}")
        return output_path

    print(f"{HOURGLASS_EMOJII} Downloading: {url}")

    with yt_dlp.YoutubeDL(
        {
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4",
            "outtmpl": f"{MEDIA_PATH}/%(id)s.%(ext)s",
        }
    ) as ydl:
        # rest of the code    ) as ydl:
        info_dict = ydl.extract_info(url)
        dl_video_id = info_dict.get("id", None)
        if dl_video_id != video_id:
            print(f"WARNING: video_id mismatch: {dl_video_id} != {video_id}")
            video_id = dl_video_id
        dl_output_path = Path(info_dict["requested_downloads"][0]["filename"])
        if dl_output_path != output_path:
            print(f"WARNING: output_file mismatch: {dl_output_path} != {output_path}")
            output_path = dl_output_path
        info_path = Path(MEDIA_PATH).joinpath(f"{video_id}_info.json")
        with open(info_path, "w") as f:
            json.dump(ydl.sanitize_info(info_dict), f, indent=2)

    print(f"{CHECK_EMOJII} Downloaded: {output_path}")
    return output_path


def extract_audio(
    video_path: Path,
    tmp_path: Path,
) -> Path:
    """
    Extract the audio stream from an input video file and save it as a WAV file.

    Args:
        video_path (Path): The path to the input video file.
        tmp_path (Path): The path to the temporary directory to store intermediate files.

    Returns:
        Path: The path to the output WAV file.

    Raises:
        FileNotFoundError: If the input video file is not found.
        ValueError: If the input video file is not a valid video file.

    Example:
        >>> extract_audio(Path('/path/to/video.mp4'), Path('/path/to/tmp'))
        Path('/path/to/tmp/video.wav')
    """
    output_path = tmp_path.joinpath(video_path.stem + ".wav")
    if output_path.exists():
        print(f"{CHECK_EMOJII} Aleady extracted: {output_path}")
        return output_path

    print(f"{HOURGLASS_EMOJII} Extracting: {video_path}")
    (
        ffmpeg.input(str(video_path))
        .output(
            str(output_path),
            format="wav",
            acodec="pcm_s16le",
            ar="44100",
            ac=2,
        )
        .overwrite_output()
        .run()
    )
    print(f"{CHECK_EMOJII} Extracted: {output_path}")
    return output_path


def add_audio_spacer(
    audio_path: Path,
    tmp_path: Path,
    milli: int = 5000,
) -> Path:
    """
    Add a silent audio spacer to the beginning of an input WAV file.

    Args:
        audio_path (Path): The path to the input audio (WAV) file.
        tmp_path (Path): The path to the temporary directory to store intermediate files.
        milli (int, optional): The duration of the spacer in milliseconds. Defaults to 5000.

    Returns:
        Path: The path to the output WAV file with the spacer added.

    Raises:
        FileNotFoundError: If the input audio file is not found.

    Example:
        >>> add_audio_spacer(Path('/path/to/input.wav'), Path('/path/to/tmp'), milli=10000)
        Path('/path/to/tmp/input_spacer.wav')
    """
    output_path = tmp_path.joinpath(audio_path.stem + "_spacer.wav")
    if output_path.exists():
        print(f"{CHECK_EMOJII} Aleady added spacer: {output_path}")
        return output_path
    print(f"{HOURGLASS_EMOJII} Adding spacer: {audio_path}")
    spacer = AudioSegment.silent(duration=milli)
    audio_path = AudioSegment.from_wav(audio_path)
    audio_path = spacer.append(audio_path, crossfade=0)
    audio_path.export(output_path, format="wav")
    print(f"{CHECK_EMOJII} Added spacer: {output_path}")
    return output_path


def diarize_audio(
    audio_path: Path,
    tmp_path: Path,
) -> Path:
    """
    Perform speaker diarization on an input WAV file using the pyannote.speaker-diarization model.

    Args:
        audio_path (Path): The path to the input audio (WAV) file.
        tmp_path (Path): The path to the temporary directory to store intermediate files.

    Returns:
        Path: The path to the output diarization file.

    Raises:
        FileNotFoundError: If the input audio file is not found.

    Example:
        >>> diarize_audio(Path('/path/to/input.wav'), Path('/path/to/tmp'))
        Path('/path/to/tmp/input_diarization.txt')
    """
    output_path = tmp_path.joinpath(audio_path.stem + "_diarization.txt")
    if output_path.exists():
        print(f"{CHECK_EMOJII} Aleady diarized: {output_path}")
        return output_path
    print(f"{HOURGLASS_EMOJII} Diarizing: {audio_path}")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization@2.1",
        use_auth_token=(HUGGINGFACE_ACCESS_TOKEN) or True,
    )
    dz = pipeline(audio_path)
    with open(output_path, "w") as text_file:
        text_file.write(str(dz))
    print(f"{CHECK_EMOJII} Diarized: {output_path}")
    return output_path


def ms_from_time_string(
    timeStr: str,
) -> int:
    """
    Convert a time string in the format 'HH:MM:SS.sss' to milliseconds.

    Args:
        timeStr (str): A string representing a time in the format 'HH:MM:SS.sss'.

    Returns:
        int: The time in milliseconds.

    Example:
        >>> ms_from_time_string('01:23:45.678')
        5025678
    """
    spl = timeStr.split(":")
    s = (int)((int(spl[0]) * 60 * 60 + int(spl[1]) * 60 + float(spl[2])) * 1000)
    return s


def group_by_speaker(
    diarization: Path,
) -> list[list[str]]:
    """
    Group consecutive segments in a speaker diarization file by speaker.

    Args:
        diarization (Path): The path to the speaker diarization file.

    Returns:
        list[list[str]]: A list of lists, where each inner list contains the segments for a single speaker.

    Example:
        >>> group_by_speaker('/path/to/diarization.txt')
        [['speaker1 00:00:00.000 --> 00:00:05.000', 'speaker1 00:00:10.000 --> 00:00:15.000'], ['speaker2 00:00:05.000 --> 00:00:10.000']]
    """
    dzs = open(diarization).read().splitlines()

    groups = []
    g = []
    lastend = 0

    for d in dzs:
        if g and (g[0].split()[-1] != d.split()[-1]):  # same speaker
            groups.append(g)
            g = []

        g.append(d)

        end = re.findall("[0-9]+:[0-9]+:[0-9]+\.[0-9]+", string=d)[1]
        end = ms_from_time_string(end)
        if lastend > end:  # segment engulfed by a previous segment
            groups.append(g)
            g = []
        else:
            lastend = end
    if g:
        groups.append(g)
    return groups


def split_audio_by_speaker(
    audio_path: Path,
    diarization_path: Path,
    tmp_path: Path,
) -> Path:
    """
    Split an input audio file into separate WAV files for each speaker group.

    Args:
        audio_path (Path): The path to the input audio (WAV) file.
        diarization_path (Path): The path to the speaker diarization file.
        tmp_path (Path): The path to the temporary directory to store intermediate files.

    Returns:
        Path: The path to the directory containing the split WAV files.

    Raises:
        FileNotFoundError: If either the input audio or diarization file is not found.

    Example:
        >>> split_audio_by_speaker(Path('/path/to/input.wav'), Path('/path/to/diarization.txt'), Path('/path/to/tmp'))
        Path('/path/to/tmp/splits')
    """
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not diarization_path.is_file():
        raise FileNotFoundError(f"Diarization file not found: {diarization_path}")

    splits_path = tmp_path.joinpath("splits")
    splits_path.mkdir(parents=True, exist_ok=True)
    speaker_groups = group_by_speaker(diarization_path)
    audio_seg = AudioSegment.from_wav(audio_path)
    gidx = -1
    for g in speaker_groups:
        start = re.findall("[0-9]+:[0-9]+:[0-9]+\.[0-9]+", string=g[0])[0]
        end = re.findall("[0-9]+:[0-9]+:[0-9]+\.[0-9]+", string=g[-1])[1]
        start = ms_from_time_string(start)
        end = ms_from_time_string(end)
        gidx += 1
        file = splits_path.joinpath(f"{gidx:03d}.wav")
        if file.exists():
            print(f"{CHECK_EMOJII} Aleady split: {file}")
            continue
        audio_seg[start:end].export(file, format="wav")
        print(f"group {gidx}: {start}--{end}")
    return splits_path


def load_whisper_model() -> whisper.Whisper:
    print(f"{HOURGLASS_EMOJII} Loading transcription model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = whisper.load_model("large", device=device)
    return model


def transcribe_audio_file(
    audio_path: Path,
) -> Path:
    """
    Transcribe a single WAV file using the whisper speech recognition model.

    Args:
        wav_path (Path): The path to the input WAV file.

    Returns:
        Dict[str, Any]: A dictionary containing the transcript and word timestamps.

    Raises:
        FileNotFoundError: If the input file is not found.
    """

    if not audio_path.exists():
        raise FileNotFoundError(f"File not found: {audio_path}")

    transcript_path = Path(TRANSCRIPTS_PATH).joinpath(audio_path.parent.stem + ".json")
    if transcript_path.exists():
        print(f"{CHECK_EMOJII} Aleady transcribed: {transcript_path}")
        return transcript_path

    output = Path().joinpath(audio_path.parent, audio_path.stem + ".json")
    if output.exists():
        print(f"{CHECK_EMOJII} Aleady transcribed: {output}")
        result = json.load(open(output))
    else:
        model = load_whisper_model()
        print(f"{HOURGLASS_EMOJII} Transcribing: {audio_path}")
        result = model.transcribe(
            audio=str(audio_path),
            language="en",
            word_timestamps=True,
        )
        with open(output, "w") as outfile:
            json.dump(result, outfile, indent=2)

    transcript = [
        {
            "speaker": 0,
            "text": result["text"],
        }
    ]
    with open(transcript_path, "w") as outfile:
        json.dump(transcript, outfile, indent=2)

    return transcript_path


def transcribe_audio_files(
    split_path: Path,
) -> Path:
    """
    Transcribe all WAV files in a given directory using the whisper speech recognition model.

    Args:
        split_path (Path): The path to the directory containing the input WAV files.

    Returns:
        Path: The path to the transcript.

    Raises:
        FileNotFoundError: If the input directory is not found.

    Example:
        >>> transcribe_audio_files(Path('/path/to/tmp'))
    """

    transcript_path = Path(TRANSCRIPTS_PATH).joinpath(split_path.parent.stem + ".json")
    if transcript_path.exists():
        print(f"{CHECK_EMOJII} Aleady transcribed: {transcript_path}")
        return transcript_path

    model = load_whisper_model()

    transcript = []
    speaker = 0  # toggles between 0 and 1
    wav_files = list(split_path.glob("*.wav"))
    wav_files.sort()
    for file in wav_files:
        output = Path().joinpath(file.parent, file.stem + ".json")
        if output.exists():
            print(f"{CHECK_EMOJII} Aleady transcribed: {output}")
            result = json.load(open(output))
        else:
            print(f"{HOURGLASS_EMOJII} Transcribing: {file}")
            result = model.transcribe(
                audio=str(file),
                language="en",
                word_timestamps=True,
            )
        transcript.append({"speaker": speaker, "text": result["text"]})
        speaker = 1 - speaker
        with open(output, "w") as outfile:
            json.dump(result, outfile, indent=2)

    with open(transcript_path, "w") as outfile:
        json.dump(transcript, outfile, indent=2)

    return transcript_path


def transcribe_audio(
    audio_path: Path,
    tmp_path: Path,
    diarize: bool = False,
) -> Path:
    """
    Transcribe an input audio file by performing speaker diarization, splitting the audio into separate WAV files for each speaker group, and transcribing each speaker's audio using the whisper speech recognition model.

    Args:
        audio_path (Path): The path to the input audio (WAV) file.
        tmp_path (Path): The path to the temporary directory to store intermediate files.

    Returns:
        Path: The path to the output transcript file.

    Raises:
        FileNotFoundError: If the input audio file is not found.

    Example:
        >>> transcribe_audio(Path('/path/to/input.wav'))
        Path('/path/to/transcripts/input.json')
    """
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if diarize:
        audio_path = add_audio_spacer(audio_path, tmp_path)
        diariazation_path = diarize_audio(audio_path, tmp_path)
        splits_path = split_audio_by_speaker(audio_path, diariazation_path, tmp_path)
        transcript_path = transcribe_audio_files(splits_path)
    else:
        transcript_path = transcribe_audio_file(audio_path)
    return transcript_path


def transcribe_media(
    input_media: str,
    diarize: bool = False,
) -> Path:
    """
    Transcribe the audio of a video file or audio file using the whisper speech recognition model.

    Args:
        input_media (str): The path to the input audio/video file or YouTube URL.
        diarize (bool, optional): Whether to perform speaker diarization. Defaults to False.

    Returns:
        Path: The path to the output transcript file.

    Raises:
        ValueError: If neither a YouTube URL nor an input file path is provided.

    Example:
        >>> transcribe_media(input_media='/path/to/file.mp4', diarize=True)
        /path/to/file.json
    """
    if input_media.startswith("http"):
        input_path = download_youtube(input_media)
    else:
        input_path = Path(input_media)

    tmp_path = Path(MEDIA_PATH).joinpath(input_path.stem)
    tmp_path.mkdir(parents=True, exist_ok=True)
    print(f"{HOURGLASS_EMOJII} Transcribing: {input_path}")
    audio_path = extract_audio(input_path, tmp_path)
    transcript_path = transcribe_audio(audio_path, tmp_path, diarize)
    print(f"{CHECK_EMOJII} Transcribed: {transcript_path}")
    return transcript_path


# if __name__ == "__main__":
#     input_video = "media/ryan.mp4"
#     youtube_url = "https://youtu.be/NSp2fEQ6wyA"

#     transcribe_video(
#         youtube_url=youtube_url,
#         # input_video=input_video,
#     )

from app import transcribe_media

app = typer.Typer()


@app.command()
def transcribe(
    input_path: str = typer.Argument(
        ..., help="Path to the input video file or YouTube URL"
    ),
    diarize: bool = typer.Option(
        False,
        "--diarize",
        help="Diarize (separate speakers) the audio/video.",
    ),
) -> None:
    """
    Transcribe an audio or video file with optional diarization.

    Args:
        input_path (str): Path to the input video file or YouTube URL.
        diarize (bool): Whether to diarize (separate speakers) the audio/video.
    """
    if input_path.startswith("http"):
        transcribe_media(
            youtube_url=input_path,
            diarize=diarize,
        )
    else:
        transcribe_media(
            input_video=input_path,
            diarize=diarize,
        )


if __name__ == "__main__":
    app()
