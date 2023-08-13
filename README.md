# Transcriber

Transcriber is a command line tool for transcribing audio and video files, with optional diarization, using the [Whisper]() speech recognition model. It supports both local files and YouTube URLs.

## Usage

To transcribe a local audio or video file:

```sh
python app.py /path/to/file.mp4
```

To transcribe a YouTube video:

```sh
python app.py https://www.youtube.com/watch?v=dQw4w9WgXcQ
```

By default, Transcriber will not use diarization to separate speakers. You can enable diarization using the `--diarize` option:

```sh
python app.py /path/to/file.mp4 --diarize
```

## License

Transcriber is licensed under the MIT License. See the LICENSE file for more information.
