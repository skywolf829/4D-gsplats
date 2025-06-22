from pathlib import Path
import cv2
import torch
import numpy as np
import gradio as gr
import shutil
from datetime import datetime
import glob
import gc
import time
import os

from dynamic_gsplats.scene_init import init_model, model_inference, DEVICE
from dynamic_gsplats.file_ops import load_images_from_dir, resize_images_to_similar_square, extract_frames_parallel
from dynamic_gsplats.visuals_utils import predictions_to_glb
from dynamic_gsplats.gsplat_train import Runner, Config

# Update examples to use time-varying data
cut_roasted_beef_videos = "examples/time_varying/cut_roasted_beef"

model = init_model()

def update_gallery_on_upload(input_videos, input_images):
    """
    Whenever user uploads or changes files, immediately handle them
    and show in the gallery. Return (target_dir, image_paths, max_timesteps).
    If nothing is uploaded, returns "None" and empty list.
    """
    if not input_videos and not input_images:
        return None, None, None, None, None, gr.Slider(minimum=0, maximum=0, value=0, visible=False)
    target_dir, image_paths, max_timesteps = handle_uploads(input_videos, input_images, timestep=0)
    
    # Show timestep slider if we have videos
    if input_videos and max_timesteps > 1:
        timestep_slider = gr.Slider(
            minimum=0, 
            maximum=max_timesteps-1, 
            value=0, 
            step=1,
            label="Timestep (Frame)",
            visible=True,
            interactive=True
        )
    else:
        timestep_slider = gr.Slider(minimum=0, maximum=0, value=0, visible=False)
    
    return None, target_dir, image_paths, "Upload complete. Click 'Reconstruct' to begin 3D processing.", max_timesteps, timestep_slider

def update_gallery_on_timestep(target_dir, timestep, max_timesteps):
    """
    When timestep slider changes, re-extract frames from videos at that timestep.
    Uses parallel processing for efficiency.
    """
    if not target_dir or target_dir == "None":
        return None, "No time-varying data loaded."
    
    try:
        max_ts = int(max_timesteps)
    except (ValueError, TypeError):
        max_ts = 1
    
    if max_ts <= 1:
        return None, "No time-varying data loaded."
    
    # Re-extract frames at the new timestep
    video_dir = Path(target_dir) / "videos"
    if not video_dir.exists():
        return None, "No videos found."
    
    target_dir_images = Path(target_dir) / "images"
    # Clear existing images
    if target_dir_images.exists():
        shutil.rmtree(target_dir_images)
    target_dir_images.mkdir(parents=True, exist_ok=True)
    
    video_files = sorted(video_dir.glob("*.mp4"))
    
    # Extract frames in parallel using file_ops function
    image_paths = extract_frames_parallel(video_files, timestep, target_dir_images)
    
    return image_paths, f"Updated to timestep {timestep}"

def clear_fields():
    """
    Clears the 3D viewer, the stored target_dir, and empties the gallery.
    """
    return None

def update_log():
    """
    Display a quick log message while waiting.
    """
    return "Loading and Reconstructing..."

def handle_uploads(input_videos, input_images, timestep=0):
    """
    Create a new 'target_dir' + 'images' subfolder, and place user-uploaded
    images or extracted frames from videos into it. 
    For multiple videos, treat them as synchronized multi-camera videos.
    Return (target_dir, image_paths, max_timesteps).
    """
    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    # Create a unique folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target_dir = Path("input_images") / f"{timestamp}"
    target_dir_images = target_dir / "images"
    target_dir_videos = target_dir / "videos"

    # Clean up if somehow that folder already exists
    if Path(target_dir).exists():
        shutil.rmtree(target_dir)
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    Path(target_dir_images).mkdir(parents=True, exist_ok=True)

    image_paths = []
    max_timesteps = 1

    # --- Handle images (static case) ---
    if input_images is not None:
        for file_data in input_images:
            if isinstance(file_data, dict) and "name" in file_data:
                file_path = file_data["name"]
            else:
                file_path = str(file_data)
            dst_path = Path(target_dir_images) / Path(file_path).name
            shutil.copy(file_path, dst_path)
            image_paths.append(dst_path)

    # --- Handle videos (time-varying case) ---
    if input_videos is not None:
        Path(target_dir_videos).mkdir(parents=True, exist_ok=True)
        
        # First, copy all videos and determine max timesteps
        video_files = []
        for file_data in input_videos:
            if isinstance(file_data, dict) and "name" in file_data:
                video_path = file_data["name"]
            else:
                video_path = str(file_data)
            
            # Copy video to target directory
            dst_video_path = Path(target_dir_videos) / Path(video_path).name
            shutil.copy(video_path, dst_video_path)
            video_files.append(dst_video_path)
            
            # Check frame count to determine max timesteps
            vs = cv2.VideoCapture(str(video_path))
            frame_count = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
            vs.release()
            max_timesteps = max(max_timesteps, frame_count)
        
        # Extract frames from all videos at the specified timestep (in parallel)
        image_paths.extend(extract_frames_parallel(video_files, timestep, target_dir_images))

    # Sort final images for gallery
    image_paths = sorted(image_paths)

    end_time = time.time()
    print(f"Files processed to {target_dir_images}; took {end_time - start_time:.3f} seconds")
    return target_dir, image_paths, max_timesteps

def gradio_demo(
    target_dir,
    conf_thres=3.0,
    frame_filter="All",
    mask_black_bg=False,
    mask_white_bg=False,
    show_cam=True,
    mask_sky=False,
    prediction_mode="Pointmap Regression",
):
    """
    Perform reconstruction using the already-created target_dir/images.
    """
    if not Path(target_dir).is_dir() or target_dir == "None":
        return None, "No valid target directory found. Please upload first.", None

    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    # Prepare frame_filter dropdown
    target_dir_images = Path(target_dir) / "images"
    all_files = sorted(target_dir_images.iterdir()) if target_dir_images.is_dir() else []
    all_files = [f"{i}: {filename}" for i, filename in enumerate(all_files)]
    frame_filter_choices = ["All"] + all_files

    print("Running run_model...")
    with torch.no_grad():
        images = load_images_from_dir(target_dir_images)
        images = resize_images_to_similar_square(images)
        images = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2)
        images = images.to(DEVICE)
        print(f"Images loaded and resized to {images.shape}")
        predictions = model_inference(model, images)

    # Save predictions
    prediction_save_path = Path(target_dir) / "predictions.npz"
    np.savez(prediction_save_path, **predictions)

    # Handle None frame_filter
    if frame_filter is None:
        frame_filter = "All"

    # Build a GLB file name
    glbfile = Path(target_dir) / f"glbscene_{conf_thres}_{frame_filter.replace('.', '_').replace(':', '').replace(' ', '_')}_maskb{mask_black_bg}_maskw{mask_white_bg}_cam{show_cam}_sky{mask_sky}_pred{prediction_mode.replace(' ', '_')}.glb"

    # Convert predictions to GLB
    glbscene = predictions_to_glb(
        predictions,
        conf_thres=conf_thres,
        filter_by_frames=frame_filter,
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
        show_cam=show_cam,
        mask_sky=mask_sky,
        target_dir=target_dir,
        prediction_mode=prediction_mode,
    )
    glbscene.export(file_obj=glbfile)

    # Cleanup
    del predictions
    gc.collect()
    torch.cuda.empty_cache()

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds (including IO)")
    log_msg = f"Reconstruction Success ({len(all_files)} frames). Waiting for visualization."

    return glbfile, log_msg, gr.Dropdown(choices=frame_filter_choices, value=frame_filter, interactive=True)

def update_visualization(
    target_dir, conf_thres, frame_filter, mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode, is_example
):
    """
    Reload saved predictions from npz, create (or reuse) the GLB for new parameters,
    and return it for the 3D viewer. If is_example == "True", skip.
    """

    # If it's an example click, skip as requested
    if is_example == "True":
        return None, "No reconstruction available. Please click the Reconstruct button first."

    if not target_dir or target_dir == "None" or not Path(target_dir).is_dir():
        return None, "No reconstruction available. Please click the Reconstruct button first."

    predictions_path = Path(target_dir) / "predictions.npz"
    if not predictions_path.exists():
        return None, f"No reconstruction available at {predictions_path}. Please run 'Reconstruct' first."

    key_list = [
        "pose_enc",
        "depth",
        "depth_conf",
        "world_points",
        "world_points_conf",
        "images",
        "extrinsic",
        "intrinsic",
        "world_points_from_depth",
    ]

    loaded = np.load(predictions_path)
    predictions = {key: np.array(loaded[key]) for key in key_list}

    glbfile = Path(target_dir) / f"glbscene_{conf_thres}_{frame_filter.replace('.', '_').replace(':', '').replace(' ', '_')}_maskb{mask_black_bg}_maskw{mask_white_bg}_cam{show_cam}_sky{mask_sky}_pred{prediction_mode.replace(' ', '_')}.glb"

    if not Path(glbfile).exists():
        glbscene = predictions_to_glb(
            predictions,
            conf_thres=conf_thres,
            filter_by_frames=frame_filter,
            mask_black_bg=mask_black_bg,
            mask_white_bg=mask_white_bg,
            show_cam=show_cam,
            mask_sky=mask_sky,
            target_dir=target_dir,
            prediction_mode=prediction_mode,
        )
        glbscene.export(file_obj=glbfile)

    return glbfile, "Updating Visualization"

def filter_points_and_colors(preds, conf_thres=50.0, max_points=100000, mask_black_bg=False, mask_white_bg=False):
    """
    Replicates the filtering logic from predictions_to_glb for points and colors.
    Returns filtered_points, filtered_colors, extrinsics, intrinsics, image_paths
    """
    # 1. Get points and confidence
    if "world_points" in preds:
        points = np.array(preds["world_points"])  # (N, 3) or (S, H, W, 3)
        conf = np.array(preds.get("world_points_conf", np.ones(points.shape[:-1])))
    else:
        points = np.array(preds["world_points_from_depth"])  # fallback
        conf = np.array(preds.get("depth_conf", np.ones(points.shape[:-1])))
    # 2. Get images
    images = np.array(preds["images"])  # (N, H, W, 3) or (N, 3, H, W)
    # 3. Flatten points, conf, and images
    points = points.reshape(-1, 3)
    conf = conf.reshape(-1)
    # Handle image format
    if images.ndim == 4 and images.shape[1] == 3:
        # NCHW -> NHWC
        images = np.transpose(images, (0, 2, 3, 1))
    colors = images.reshape(-1, 3)
    colors = (colors * 255).astype(np.uint8)
    # 4. Confidence mask
    if conf_thres == 0.0:
        conf_threshold = 0.0
    else:
        conf_threshold = np.percentile(conf, conf_thres)
    conf_mask = (conf >= conf_threshold) & (conf > 1e-5)
    # 5. Background masks
    if mask_black_bg:
        black_bg_mask = colors.sum(axis=1) >= 16
        conf_mask = conf_mask & black_bg_mask
    if mask_white_bg:
        white_bg_mask = ~((colors[:, 0] > 240) & (colors[:, 1] > 240) & (colors[:, 2] > 240))
        conf_mask = conf_mask & white_bg_mask
    # 6. Apply mask
    filtered_points = points[conf_mask]
    filtered_colors = colors[conf_mask]

    # 7. Sort and cap max_points by confidence
    if max_points is not None and len(filtered_points) > max_points:
        # Get the confidence values for the masked points
        masked_conf = conf[conf_mask]
        topk_indices = np.argsort(-masked_conf)[:max_points]  # descending sort
        filtered_points = filtered_points[topk_indices]
        filtered_colors = filtered_colors[topk_indices]

    return filtered_points, filtered_colors

def run_gsplat_refinement(target_dir):
    """
    Loads point cloud, images, and camera parameters from target_dir,
    runs GSplat training, and returns status.
    """
    status = "Starting GSplat refinement..."
    try:
        # 1. Load predictions.npz
        pred_path = Path(target_dir) / "predictions.npz"
        if not pred_path.exists():
            return gr.update(value="No predictions.npz found!")
        preds = np.load(pred_path)
        # 2. Extract camera params
        extrinsics = np.array(preds["extrinsic"]).astype(np.float32) # (N, 4, 4)
        intrinsics = np.array(preds["intrinsic"]).astype(np.float32) # (N, 3, 3)
        original_image_shape = np.array(preds["images"]).shape[1:]
        # 3. Load images from target_dir/images (sorted)
        image_dir = Path(target_dir) / "images"
        image_paths = sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpg"))
        if len(image_paths) == 0:
            return gr.update(value="No images found in target_dir/images!")
        # 4. Filter points and colors as in predictions_to_glb
        points, rgbs = filter_points_and_colors(preds, conf_thres=50.0, max_points=100000, 
            mask_black_bg=False, mask_white_bg=False)
        if len(points) == 0 or len(rgbs) == 0:
            return gr.update(value="No points/colors after filtering!")
        # 5. Run GSplat training (headless)
        cfg = Config()
        cfg.result_dir = str(Path(target_dir) / "gsplat_results")
        os.makedirs(cfg.result_dir, exist_ok=True)
        runner = Runner(
            init_points=points,
            init_rgb=rgbs,
            training_images=image_paths,
            training_extrinsics=extrinsics,
            training_intrinsics=intrinsics,
            local_rank=0,
            world_rank=0,
            world_size=1,
            cfg=cfg,
        )
        runner.train()  # This will run the training loop
        status = "Done!"
        return gr.update(value=status)
    except Exception as e:
        return gr.update(value=f"Error: {e}")

def main():
    theme = gr.Theme()
    theme.set(
        checkbox_label_background_fill_selected="*button_primary_background_fill",
        checkbox_label_text_color_selected="*button_primary_text_color",
    )

    with gr.Blocks(
        theme=theme,
        css="""
        .custom-log * {
            font-style: italic;
            font-size: 22px !important;
            background-image: linear-gradient(120deg, #0ea5e9 0%, #6ee7b7 60%, #34d399 100%);
            -webkit-background-clip: text;
            background-clip: text;
            font-weight: bold !important;
            color: transparent !important;
            text-align: center !important;
        }
        
        .example-log * {
            font-style: italic;
            font-size: 16px !important;
            background-image: linear-gradient(120deg, #0ea5e9 0%, #6ee7b7 60%, #34d399 100%);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent !important;
        }
        
        #my_radio .wrap {
            display: flex;
            flex-wrap: nowrap;
            justify-content: center;
            align-items: center;
        }

        #my_radio .wrap label {
            display: flex;
            width: 50%;
            justify-content: center;
            align-items: center;
            margin: 0;
            padding: 10px 0;
            box-sizing: border-box;
        }
        """,
    ) as demo:
        # Hidden state variables
        is_example = gr.Textbox(label="is_example", visible=False, value="None")
        max_timesteps = gr.Textbox(label="max_timesteps", visible=False, value="1")

        gr.HTML(
            """
        <h1>🏛️ VGGT: Visual Geometry Grounded Transformer (Time-Varying)</h1>
        <p>
        <a href="https://github.com/facebookresearch/vggt">🐙 GitHub Repository</a> |
        <a href="#">Project Page</a>
        </p>

        <div style="font-size: 16px; line-height: 1.5;">
        <p>Upload synchronized videos or a set of images to create a 3D reconstruction of a time-varying scene or object. VGGT takes these images and generates a 3D point cloud, along with estimated camera poses.</p>

        <h3>Getting Started:</h3>
        <ol>
            <li><strong>Upload Your Data:</strong> 
                <ul>
                    <li><strong>For static scenes:</strong> Use "Upload Images" to provide individual images from the same timestep.</li>
                    <li><strong>For time-varying scenes:</strong> Use "Upload Videos" to provide synchronized multi-camera videos. Use the timestep slider to select which frame to reconstruct.</li>
                </ul>
            </li>
            <li><strong>Preview:</strong> Your uploaded/extracted images will appear in the gallery on the left.</li>
            <li><strong>Select Timestep (Videos only):</strong> Use the timestep slider to choose which frame from the synchronized videos to reconstruct.</li>
            <li><strong>Reconstruct:</strong> Click the "Reconstruct" button to start the 3D reconstruction process.</li>
            <li><strong>Visualize:</strong> The 3D reconstruction will appear in the viewer on the right. You can rotate, pan, and zoom to explore the model, and download the GLB file.</li>
        </ol>
        <p><strong style="color: #0ea5e9;">Please note:</strong> <span style="color: #0ea5e9; font-weight: bold;">VGGT typically reconstructs a scene in less than 1 second. However, visualizing 3D points may take tens of seconds due to third-party rendering, which are independent of VGGT's processing time. </span></p>
        </div>
        """
        )

        target_dir_output = gr.Textbox(label="Target Dir", visible=False, value="None")

        with gr.Row():
            with gr.Column(scale=2):
                input_videos = gr.File(file_count="multiple", label="Upload Videos (Synchronized Multi-Camera)", file_types=["video"], interactive=True)
                input_images = gr.File(file_count="multiple", label="Upload Images (Same Timestep)", file_types=["image"], interactive=True)
                
                timestep_slider = gr.Slider(
                    minimum=0,
                    maximum=0,
                    value=0,
                    step=1,
                    label="Timestep (Frame)",
                    visible=False,
                    interactive=True
                )

                image_gallery = gr.Gallery(
                    label="Preview",
                    columns=4,
                    height="300px",
                    show_download_button=True,
                    object_fit="contain",
                    preview=True,
                )

            with gr.Column(scale=4):
                with gr.Column():
                    gr.Markdown("**3D Reconstruction (Point Cloud and Camera Poses)**")
                    log_output = gr.Markdown(
                        "Please upload videos or images, then click Reconstruct.", elem_classes=["custom-log"]
                    )
                    reconstruction_output = gr.Model3D(height=520, zoom_speed=0.5, pan_speed=0.5)

                with gr.Row():
                    submit_btn = gr.Button("Reconstruct", scale=1, variant="primary")
                    clear_btn = gr.ClearButton(
                        [input_videos, input_images, reconstruction_output, log_output, target_dir_output, image_gallery, timestep_slider],
                        scale=1,
                    )
                # --- New GSplat refinement button and status ---
                with gr.Row():
                    gsplat_btn = gr.Button("Run GSplat Refinement", scale=1, variant="secondary")
                    gsplat_status = gr.Textbox(label="GSplat Status", value="Idle", interactive=False)

                with gr.Row():
                    prediction_mode = gr.Radio(
                        ["Depthmap and Camera Branch", "Pointmap Branch"],
                        label="Select a Prediction Mode",
                        value="Depthmap and Camera Branch",
                        scale=1,
                        elem_id="my_radio",
                    )

                with gr.Row():
                    conf_thres = gr.Slider(minimum=0, maximum=100, value=50, step=0.1, label="Confidence Threshold (%)")
                    frame_filter = gr.Dropdown(choices=["All"], value="All", label="Show Points from Frame")
                    with gr.Column():
                        show_cam = gr.Checkbox(label="Show Camera", value=True)
                        mask_sky = gr.Checkbox(label="Filter Sky", value=False)
                        mask_black_bg = gr.Checkbox(label="Filter Black Background", value=False)
                        mask_white_bg = gr.Checkbox(label="Filter White Background", value=False)

        # ---------------------- Examples section ----------------------
        
        examples = [
            # Format: [videos_display, images, timestep, conf_thres, mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode, is_example]
            [
                glob.glob("examples/time_varying/cut_roasted_beef/*.mp4"),
                None, 
                0,  # timestep
                30.0,  # conf_thres
                False,  # mask_black_bg
                False,  # mask_white_bg
                True,   # show_cam
                False,  # mask_sky
                "Depthmap and Camera Branch",  # prediction_mode
                "True"  # is_example
            ],
        ]

        def example_pipeline(
            input_videos_display,
            input_images,
            timestep,
            conf_thres,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            is_example_str,
        ):
            """
            1) Map display name to actual video files
            2) Copy example videos/images to new target_dir
            3) Extract frames at specified timestep
            4) Reconstruct
            5) Return model3D + logs + new_dir + updated dropdown + gallery + timestep_slider
            """
            
            target_dir, image_paths, max_ts = handle_uploads(input_videos_display, input_images, timestep)
            # Always use "All" for frame_filter in examples
            frame_filter = "All"
            glbfile, log_msg, dropdown = gradio_demo(
                target_dir, conf_thres, frame_filter, mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode
            )
            
            # Set up timestep slider
            if input_videos_display and max_ts > 1:
                new_timestep_slider = gr.Slider(
                    minimum=0, 
                    maximum=max_ts-1, 
                    value=timestep, 
                    step=1,
                    label="Timestep (Frame)",
                    visible=True,
                    interactive=True
                )
            else:
                new_timestep_slider = gr.Slider(minimum=0, maximum=0, value=0, visible=False)
            
            return glbfile, log_msg, target_dir, dropdown, image_paths, new_timestep_slider

        gr.Markdown("Click any row to load an example.", elem_classes=["example-log"])

        gr.Examples(
            examples=examples,
            inputs=[
                input_videos,
                input_images,
                timestep_slider,
                conf_thres,
                mask_black_bg,
                mask_white_bg,
                show_cam,
                mask_sky,
                prediction_mode,
                is_example,
            ],
            outputs=[reconstruction_output, log_output, target_dir_output, frame_filter, image_gallery, timestep_slider],
            fn=example_pipeline,
            cache_examples=False,
            examples_per_page=50,
        )

        # -------------------------------------------------------------------------
        # "Reconstruct" button logic:
        #  - Clear fields
        #  - Update log
        #  - gradio_demo(...) with the existing target_dir
        #  - Then set is_example = "False"
        # -------------------------------------------------------------------------
        submit_btn.click(fn=clear_fields, inputs=[], outputs=[reconstruction_output]).then(
            fn=update_log, inputs=[], outputs=[log_output]
        ).then(
            fn=gradio_demo,
            inputs=[
                target_dir_output,
                conf_thres,
                frame_filter,
                mask_black_bg,
                mask_white_bg,
                show_cam,
                mask_sky,
                prediction_mode,
            ],
            outputs=[reconstruction_output, log_output, frame_filter],
        ).then(
            fn=lambda: "False", inputs=[], outputs=[is_example]  # set is_example to "False"
        )

        # --- GSplat refinement button logic ---
        gsplat_btn.click(
            fn=run_gsplat_refinement,
            inputs=[target_dir_output],
            outputs=[gsplat_status],
        )

        # -------------------------------------------------------------------------
        # Real-time Visualization Updates
        # -------------------------------------------------------------------------
        conf_thres.change(
            update_visualization,
            [
                target_dir_output,
                conf_thres,
                frame_filter,
                mask_black_bg,
                mask_white_bg,
                show_cam,
                mask_sky,
                prediction_mode,
                is_example,
            ],
            [reconstruction_output, log_output],
        )
        frame_filter.change(
            update_visualization,
            [
                target_dir_output,
                conf_thres,
                frame_filter,
                mask_black_bg,
                mask_white_bg,
                show_cam,
                mask_sky,
                prediction_mode,
                is_example,
            ],
            [reconstruction_output, log_output],
        )
        mask_black_bg.change(
            update_visualization,
            [
                target_dir_output,
                conf_thres,
                frame_filter,
                mask_black_bg,
                mask_white_bg,
                show_cam,
                mask_sky,
                prediction_mode,
                is_example,
            ],
            [reconstruction_output, log_output],
        )
        mask_white_bg.change(
            update_visualization,
            [
                target_dir_output,
                conf_thres,
                frame_filter,
                mask_black_bg,
                mask_white_bg,
                show_cam,
                mask_sky,
                prediction_mode,
                is_example,
            ],
            [reconstruction_output, log_output],
        )
        show_cam.change(
            update_visualization,
            [
                target_dir_output,
                conf_thres,
                frame_filter,
                mask_black_bg,
                mask_white_bg,
                show_cam,
                mask_sky,
                prediction_mode,
                is_example,
            ],
            [reconstruction_output, log_output],
        )
        mask_sky.change(
            update_visualization,
            [
                target_dir_output,
                conf_thres,
                frame_filter,
                mask_black_bg,
                mask_white_bg,
                show_cam,
                mask_sky,
                prediction_mode,
                is_example,
            ],
            [reconstruction_output, log_output],
        )
        prediction_mode.change(
            update_visualization,
            [
                target_dir_output,
                conf_thres,
                frame_filter,
                mask_black_bg,
                mask_white_bg,
                show_cam,
                mask_sky,
                prediction_mode,
                is_example,
            ],
            [reconstruction_output, log_output],
        )

        # -------------------------------------------------------------------------
        # Auto-update gallery whenever user uploads or changes their files
        # -------------------------------------------------------------------------
        input_videos.change(
            fn=update_gallery_on_upload,
            inputs=[input_videos, input_images],
            outputs=[reconstruction_output, target_dir_output, image_gallery, log_output, max_timesteps, timestep_slider],
        )
        input_images.change(
            fn=update_gallery_on_upload,
            inputs=[input_videos, input_images],
            outputs=[reconstruction_output, target_dir_output, image_gallery, log_output, max_timesteps, timestep_slider],
        )

        # -------------------------------------------------------------------------
        # Timestep slider change handler
        # -------------------------------------------------------------------------
        timestep_slider.change(
            fn=update_gallery_on_timestep,
            inputs=[target_dir_output, timestep_slider, max_timesteps],
            outputs=[image_gallery, log_output],
        )

        demo.queue(max_size=20).launch(show_error=True, share=False)

if __name__ == "__main__":
    main()