from pathlib import Path
from collections import defaultdict
import subprocess
import cv2


def extract_frames(video_path, output_dir='./tmp_data', fps_target=2):
    """
    Extracts frames from a video at the specified frame rate.
    
    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory where extracted frames will be saved.
        fps_target (int): Frames per second to extract.
    
    Returns:
        List[Path]: List of paths to the extracted frame image files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_pattern = str(output_dir / "frame_%05d.jpg")
    
    # FFmpeg command
    extract_cmd = [
        'ffmpeg',
        '-y',
        '-i', video_path,
        '-vf', f'fps={fps_target}',
        '-q:v', '2',
        frame_pattern
    ]
    
    try:
        print('Running FFmpeg to extract frames...')
        subprocess.run(extract_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print('FFmpeg extraction completed.')
    except subprocess.CalledProcessError as e:
        print('FFmpeg failed, using OpenCV fallback...')
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 1
        interval = int(fps / fps_target) if fps >= fps_target else 1
        count, extracted = 0, 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if count % interval == 0:
                frame_path = output_dir / f"frame_{extracted:05d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                extracted += 1
            count += 1
        cap.release()
        print(f"Extracted {extracted} frames via OpenCV to {output_dir}")

    # Return sorted list of frame paths
    frame_files = sorted(output_dir.glob('frame_*.jpg'))
    print(f"Total extracted frames: {len(frame_files)} (location: {output_dir})")
    return frame_files


def extract_gifs_from_video(video_path, timeframes, output_dir='./gifs', scale_width=None, fps=10):
    """
    Extracts GIFs from specific timeframes in a video.
    
    Args:
        video_path (str): Path to the input video.
        timeframes (List[Tuple[str, str]]): List of (start_time, end_time) tuples in HH:MM:SS or seconds format.
        output_dir (str): Directory to save the generated GIFs.
        scale_width (int): Optional. Resize GIF to this width while preserving aspect ratio.
        fps (int): Frame rate for the output GIFs.

    Returns:
        List[Path]: Paths to generated GIFs.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    gif_paths = []
    
    for idx, (start, end) in enumerate(timeframes):
        gif_path = output_dir / f'clip_{idx:02d}.gif'
        duration = f'{float(end) - float(start)}' if start.replace('.', '', 1).isdigit() else None

        # Filter string for optional resizing
        filters = [f'fps={fps}']
        if scale_width:
            filters.append(f'scale={scale_width}:-1')  # maintain aspect ratio
        filter_str = ",".join(filters)

        cmd = [
            'ffmpeg',
            '-y',
            '-ss', str(start),
        ]
        if duration:
            cmd += ['-t', duration]
        else:
            cmd += ['-to', str(end)]

        cmd += [
            '-i', video_path,
            '-vf', filter_str,
            '-loop', '0',  # do not loop forever
            str(gif_path)
        ]
        
        try:
            print(f"Creating GIF for clip {idx} from {start} to {end}...")
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            gif_paths.append(gif_path)
        except subprocess.CalledProcessError as e:
            print(f"Failed to create GIF for clip {idx}: {e.stderr.decode('utf-8')}")
    
    print(f"Generated {len(gif_paths)} GIFs in {output_dir}")
    return gif_paths

def extract_video_clips(video_path, timeframes, output_dir='./clips', output_format='mp4'):
    """
    Extracts video clips from specific timeframes in a video.

    Args:
        video_path (str): Path to the input video.
        timeframes (List[Tuple[str, str]]): List of (start_time, end_time) tuples.
        output_dir (str): Directory to save extracted clips.
        output_format (str): Output video format (e.g., 'mp4', 'mov', 'mkv').

    Returns:
        List[Path]: List of paths to the extracted video clips.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clip_paths = []

    for idx, (start, end) in enumerate(timeframes):
        clip_path = output_dir / f'clip_{idx:02d}.{output_format}'
        duration = f'{float(end) - float(start)}' if start.replace('.', '', 1).isdigit() else None

        cmd = [
            'ffmpeg',
            '-y',
            '-ss', str(start),
        ]
        if duration:
            cmd += ['-t', duration]
        else:
            cmd += ['-to', str(end)]

        cmd += [
            '-i', video_path,
            '-c', 'copy',  # no re-encoding for speed
            str(clip_path)
        ]

        try:
            print(f"Extracting clip {idx} from {start} to {end}...")
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            clip_paths.append(clip_path)
        except subprocess.CalledProcessError as e:
            print(f"Failed to extract clip {idx}: {e.stderr.decode('utf-8')}")

    print(f"Extracted {len(clip_paths)} video clips to {output_dir}")
    return clip_paths


def get_time_slots_per_clusterings(labels, fps:int=2):
    """
    Get time intervals per cluster ID based on a stack of label vectors and FPS.

    Args:
        labels (list or np.ndarray): List of cluster labels, one per frame.
        fps (int): Frames per second.

    Returns:
        dict: {cluster_id: [(start_time, end_time), ...]}
    """
    assert len(labels.shape) == 2

    out = []

    for i in range(labels.shape[1]):
        out += [get_time_slots_per_clustering(labels[:,i], fps)]

    return out

def get_time_slots_per_clustering(labels, fps):
    """
    Get time intervals per cluster ID in hh:mm:ss:msms format based on a label vector and FPS.

    Args:
        labels (list or np.ndarray): List of cluster labels, one per frame.
        fps (int or float): Frames per second.

    Returns:
        dict: {cluster_id: [(start_time_str, end_time_str), ...]}
    """

    def seconds_to_hhmmssms(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds - int(seconds)) * 1000)
        return f"{hours:02}:{minutes:02}:{secs:02}.{milliseconds:03}"

    if fps <= 0:
        raise ValueError("FPS must be a positive number.")

    cluster_times = defaultdict(list)
    prev_label = labels[0]
    start_frame = 0

    for i in range(1, len(labels)):
        current_label = labels[i]
        if current_label != prev_label:
            # Convert frames to times
            start_time = start_frame / fps
            end_time = i / fps
            cluster_times[prev_label].append((
                seconds_to_hhmmssms(start_time),
                seconds_to_hhmmssms(end_time)
            ))
            start_frame = i
            prev_label = current_label

    # Add the last segment
    start_time = start_frame / fps
    end_time = len(labels) / fps
    cluster_times[prev_label].append((
        seconds_to_hhmmssms(start_time),
        seconds_to_hhmmssms(end_time)
    ))

    return dict(cluster_times)