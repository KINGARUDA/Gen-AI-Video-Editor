import gradio as gr
from PIL import Image
import torch
import os
import subprocess
import pathlib
import time
import subprocess
import cv2
from pytorch_reconet.reconet_inference import process_video
from PIL import Image
from tqdm import tqdm
from moviepy.editor import VideoFileClip  # For video duration
import numpy as np

def add_audio_to_video(original_video, interpolated_video, final_output):
    import subprocess
    import os

    extracted_audio = "temp_audio.aac"
    stretched_audio = None  # ✅ fix: safe default

    # Step 1: Extract original audio
    subprocess.run(
        ["ffmpeg", "-y", "-i", original_video, "-vn", "-acodec", "copy", extracted_audio],
        capture_output=True, text=True
    )

    # Step 2: Get durations
    def get_duration(file):
        import ffmpeg
        try:
            probe = ffmpeg.probe(file)
            return float(probe["format"]["duration"])
        except Exception as e:
            print(f"Error getting duration for {file}: {e}")
            return 0.0

    orig_duration = get_duration(original_video)
    interp_duration = get_duration(interpolated_video)

    # Step 3: If durations match, copy audio directly
    if abs(orig_duration - interp_duration) < 0.1:
        cmd = [
            "ffmpeg", "-y",
            "-i", interpolated_video, "-i", extracted_audio,
            "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0",
            final_output
        ]
    else:
        # Step 4: Stretch audio to match
        stretched_audio = "temp_audio_stretched.aac"
        atempo = interp_duration / orig_duration

        # sanity check for atempo limits
        if not (0.5 <= atempo <= 2.0):
            print("⚠️ Stretch factor too extreme; skipping audio.")
            return interpolated_video

        subprocess.run(
            ["ffmpeg", "-y", "-i", extracted_audio, "-filter:a", f"atempo={atempo}", stretched_audio],
            capture_output=True, text=True
        )

        cmd = [
            "ffmpeg", "-y",
            "-i", interpolated_video, "-i", stretched_audio,
            "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0",
            final_output
        ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("FFmpeg failed to add audio:", result.stderr)
        raise RuntimeError("Audio insertion failed.")

    # Step 5: Clean up
    to_delete = [extracted_audio]
    if stretched_audio and interp_duration != orig_duration:
        to_delete.append(stretched_audio)

    for f in to_delete:
        if os.path.exists(f):
            os.remove(f)

    return final_output


def trim_video(video_path, start_time, end_time):
    """
    Trim the video between start_time and end_time using FFmpeg.

    Args:
    - video_path: path to input video
    - start_time: float, start time in seconds
    - end_time: float, end time in seconds

    Returns:
    - Path to trimmed video
    """
    output_path = os.path.join(BASE_DIR, "trimmed_output.mp4")
    duration = end_time - start_time
    if duration <= 0:
        raise ValueError("End time must be greater than start time.")

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-ss", str(start_time),
        "-t", str(duration),
        "-c", "copy",
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("FFmpeg trimming failed:", result.stderr)
        raise RuntimeError("Video trimming failed.")

    return output_path

# Get absolute base directory (summer_ml_space/)
BASE_DIR = os.getcwd()
CORE_DIR = os.path.join(BASE_DIR, "core")
VIDEO_DIR = os.path.join(BASE_DIR, "Video-Inference")

def stylize_image(content_img, style_img, out_dir=None):
    content_img = pathlib.Path(content_img).expanduser().resolve()
    style_img = pathlib.Path(style_img).expanduser().resolve()

    if out_dir is None:
        out_dir = content_img.parent / "stylised"
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    output_img = out_dir / f"{content_img.stem}_stylised.jpg"

    model_path = os.path.join(CORE_DIR, "models", "vgg16-00b39a1b.pth")
    script_path = os.path.join(CORE_DIR, "neural_style.py")

    device = "0" if torch.cuda.is_available() else "-1"

    cmd = [
        "python", script_path,
        "-content_image", str(content_img),
        "-style_image", str(style_img),
        "-output_image", str(output_img),
        "-gpu", device,
        "-print_iter", "100",
        "-model_file", model_path,
        "-num_iterations", "1500",
        "-save_iter", "1500",
        "-init", "image",
        "-backend", "cudnn",
        "-cudnn_autotune",
        "-image_size", "512"
    ]

    t0 = time.time()
    subprocess.run(cmd, check=True)
    dt = time.time() - t0
    return str(output_img), f"{dt:.2f}s"

def interpolate_video(video_path, model_choice):
    output_path = os.path.join(BASE_DIR, "output.mp4")   
    model_paths = {
        "RIFE v1.5": os.path.join(VIDEO_DIR, "models", "rife_HD_1_5_converted.pth"),
        "SOFVSR_REDS_F1": os.path.join(VIDEO_DIR,"models", "SOFVSR_REDS_F1_V1.pth"),
        "VIMEO BI": os.path.join(VIDEO_DIR, "models", "TecoGAN_4x_BI_Vimeo_iter500K.pth"),
        "VIMEO BD": os.path.join(VIDEO_DIR, "models", "TecoGAN_4x_BO_Vimeo_iter500K.pth")
        # Add more models here
    }

    selected_model = model_paths[model_choice]
    run_path = os.path.join(VIDEO_DIR, "run.py")
    command = f"python \"{run_path}\" {selected_model} --input \"{video_path}\" --output \"{output_path}\""
    os.system(command)

    final_output_path = os.path.join(BASE_DIR, "final_with_audio.mp4")
    final_output_path = add_audio_to_video(video_path, output_path, final_output_path)
    return final_output_path

def run_reconet(video_input):
    RECONET_DIR = os.path.join(BASE_DIR, "pytorch_reconet")
    output_dir = os.path.join(RECONET_DIR, "results")
    os.makedirs(output_dir, exist_ok=True)

    intermediate_output = os.path.join(output_dir, "output_raw.avi")
    final_output = os.path.join(output_dir, "output.mp4")

    model_weights = os.path.join(RECONET_DIR, "model_mosaic_2.pth")

    try:
        result = process_video(video_input, intermediate_output, model_weights)
        result_with_audio = add_audio_to_video(video_input, result, final_output)
        return result_with_audio
    except Exception as e:
        print("[ERROR]", e)
        return "Error during inference"

# AnimeGAN Function (adapted from your notebook)
def animegan_process_video(input_path, style='face_paint_512_v2', size=512, fps=None,
                           device='cuda' if torch.cuda.is_available() else 'cpu',
                           output_path="anime_style_output.mp4"):
    import numpy as np  # In case it's missing
    import os

    # Load AnimeGANv2
    generator = torch.hub.load('bryandlee/animegan2-pytorch:main', 'generator', pretrained=style, device=device).eval().to(device)
    face2paint = torch.hub.load('bryandlee/animegan2-pytorch:main', 'face2paint', size=size, device=device)

    # Setup video reading and writing
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return f"Error opening video: {input_path}"

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps or source_fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    temp_no_audio = "temp_no_audio.mp4"  # intermediate output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(temp_no_audio, fourcc, fps, (width, height))

    with torch.no_grad(), tqdm(total=total_frames, desc="Stylizing", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            stylized_pil = face2paint(generator, image_pil)
            stylized_bgr = cv2.cvtColor(np.array(stylized_pil.resize((width, height), Image.LANCZOS)), cv2.COLOR_RGB2BGR)
            writer.write(stylized_bgr)
            pbar.update(1)

    cap.release()
    writer.release()

    return add_audio_to_video(input_path, temp_no_audio, output_path)

def anime_interface():
    def process_anime(input_video, style, size, fps):
        output_path = animegan_process_video(
            input_video, style=style, size=size, fps=None if fps == 0 else fps
        )
        return output_path, "Stylized successfully!"

    with gr.Blocks() as anime_block:
        gr.Markdown("## AnimeGANv2 Video Stylization")

        input_video = gr.Video(label="Upload Video")  # ← move inside the layout

        with gr.Row():
            style_dropdown = gr.Dropdown(
                choices=['face_paint_512_v2', 'paprika', 'celeba_distill', 'face_paint_512_v1'],
                label="Anime Style",
                value='face_paint_512_v2'
            )
            size_slider = gr.Slider(256, 1024, value=512, step=64, label="Processing Size")
            fps_slider = gr.Slider(0, 60, value=0, step=1, label="Output FPS (0 = keep original)")

        stylize_btn = gr.Button("Stylize Video")
        output_video = gr.Video(label="Stylized Output")
        status = gr.Textbox(label="Status")

        stylize_btn.click(
            process_anime,
            inputs=[input_video, style_dropdown, size_slider, fps_slider],
            outputs=[output_video, status]
        )

    return anime_block

# Define the Gradio interfaces for each feature
image_stylization = gr.Interface(
    fn=stylize_image,
    inputs=[
        gr.Image(type="filepath", label="Content Image"),
        gr.Image(type="filepath", label="Style Image")
    ],
    outputs=[
        gr.Image(type="filepath", label="Stylized Output"),
        gr.Textbox(label="Processing Time")
    ],
    title="Image Stylization"
)

video_interpolation = gr.Interface(
    fn=interpolate_video,
    inputs=[
        gr.Video(label="Upload Video"),
        gr.Dropdown(
            choices=["RIFE v1.5", "SOFVSR_REDS_F1", "VIMEO BI", "VIMEO BD"],
            label="Select Model"
        )
    ],
    outputs=gr.Video(label="Interpolated Output"),
    title="Video Interpolation"
)

reconet = gr.Interface(
    fn=run_reconet,
    inputs=gr.Video(label="Upload Input Video"),
    outputs=gr.Video(label="Stylized Output Video"),
    title="ReCoNet Style Transfer Test"
)

video_trimming = gr.Interface(
    fn=trim_video,
    inputs=[
        gr.Video(label="Upload Video"),
        gr.Number(label="Start Time (s)", value=0),
        gr.Number(label="End Time (s)", value=5)
    ],
    outputs=gr.Video(label="Trimmed Video"),
    title="Video Trimming"
)

# Create a tabbed interface
demo = gr.TabbedInterface(
    [image_stylization, video_interpolation, reconet, video_trimming, anime_interface()],
    ["Image Stylization", "Video Interpolation", "Reconet Stylization", "Video Trimming", "AnimeGAN Stylization"]
)

if __name__ == "__main__":
    demo.launch()

