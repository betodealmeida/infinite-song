cat 29_hour_long_song | ffmpeg -f f32le -acodec pcm_f32le -ar 24000 -ac 1 -i pipe: -f mp3 pipe: | cat > 29_hour_long_song.mp3
ffmpeg -i in.mp3 -af "afade=t=in:st=0:d=60,afade=t=out:st=2505540:d=60" out.mp3
