for f in *.mp4; do
  ffmpeg -y \
    -f lavfi -i color=c=white:s=1920x1080:r=30 \
    -i "$f" \
    -c:v libx264 -preset medium -crf 18 \
    -pix_fmt yuv420p \
    -c:a copy \
    -shortest \
    "${f%.mp4}_video.mp4"
done
