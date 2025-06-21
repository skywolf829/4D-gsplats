from pathlib import Path
import cv2
import torch
import numpy as np
import gradio as gr
import sys
import shutil
from datetime import datetime
import glob
import gc
import time

from scene_init import init_model, model_inference, DEVICE
from file_ops import load_images_from_dir, resize_images_to_similar_square
from visuals_utils import predictions_to_glb

great_wall_video = "examples/videos/great_wall.mp4"
colosseum_video = "examples/videos/Colosseum.mp4"
room_video = "examples/videos/room.mp4"
kitchen_video = "examples/videos/kitchen.mp4"
fern_video = "examples/videos/fern.mp4"
single_cartoon_video = "examples/videos/single_cartoon.mp4"
single_oil_painting_video = "examples/videos/single_oil_painting.mp4"
pyramid_video = "examples/videos/pyramid.mp4"

model = init_model()

def update_gallery_on_upload(input_video, input_images):
    """
    Whenever user uploads or changes files, immediately handle them
    and show in the gallery. Return (target_dir, image_paths).
    If nothing is uploaded, returns "None" and empty list.
    """
    if not input_video and not input_images:
        return None, None, None, None
    target_dir, image_paths = handle_uploads(input_video, input_images)
    return None, target_dir, image_paths, "Upload complete. Click 'Reconstruct' to begin 3D processing."

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

def handle_uploads(input_video, input_images):
    """
    Create a new 'target_dir' + 'images' subfolder, and place user-uploaded
    images or extracted frames from video into it. Return (target_dir, image_paths).
    """
    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    # Create a unique folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target_dir = Path("input_images") / f"{timestamp}"
    target_dir_images = target_dir / "images"

    # Clean up if somehow that folder already exists
    if Path(target_dir).exists():
        shutil.rmtree(target_dir)
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    Path(target_dir_images).mkdir(parents=True, exist_ok=True)

    image_paths = []

    # --- Handle images ---
    if input_images is not None:
        for file_data in input_images:
            if isinstance(file_data, dict) and "name" in file_data:
                file_path = file_data["name"]
            else:
                file_path = file_data
            dst_path = Path(target_dir_images) / Path(file_path).name
            shutil.copy(file_path, dst_path)
            image_paths.append(dst_path)

    # --- Handle video ---
    if input_video is not None:
        if isinstance(input_video, dict) and "name" in input_video:
            video_path = input_video["name"]
        else:
            video_path = input_video

        vs = cv2.VideoCapture(video_path)
        fps = vs.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * 1)  # 1 frame/sec

        count = 0
        video_frame_num = 0
        while True:
            gotit, frame = vs.read()
            if not gotit:
                break
            count += 1
            if count % frame_interval == 0:
                image_path = Path(target_dir_images) / f"{video_frame_num:06}.png"
                cv2.imwrite(image_path, frame)
                image_paths.append(image_path)
                video_frame_num += 1

    # Sort final images for gallery
    image_paths = sorted(image_paths)

    end_time = time.time()
    print(f"Files copied to {target_dir_images}; took {end_time - start_time:.3f} seconds")
    return target_dir, image_paths

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
        return None, "No valid target directory found. Please upload first.", None, None

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
        # Instead of gr.State, we use a hidden Textbox:
        is_example = gr.Textbox(label="is_example", visible=False, value="None")
        num_images = gr.Textbox(label="num_images", visible=False, value="None")

        gr.HTML(
            """
        <h1>🏛️ VGGT: Visual Geometry Grounded Transformer</h1>
        <p>
        <a href="https://github.com/facebookresearch/vggt">🐙 GitHub Repository</a> |
        <a href="#">Project Page</a>
        </p>

        <div style="font-size: 16px; line-height: 1.5;">
        <p>Upload a video or a set of images to create a 3D reconstruction of a scene or object. VGGT takes these images and generates a 3D point cloud, along with estimated camera poses.</p>

        <h3>Getting Started:</h3>
        <ol>
            <li><strong>Upload Your Data:</strong> Use the "Upload Video" or "Upload Images" buttons on the left to provide your input. Videos will be automatically split into individual frames (one frame per second).</li>
            <li><strong>Preview:</strong> Your uploaded images will appear in the gallery on the left.</li>
            <li><strong>Reconstruct:</strong> Click the "Reconstruct" button to start the 3D reconstruction process.</li>
            <li><strong>Visualize:</strong> The 3D reconstruction will appear in the viewer on the right. You can rotate, pan, and zoom to explore the model, and download the GLB file. Note the visualization of 3D points may be slow for a large number of input images.</li>
            <li>
            <strong>Adjust Visualization (Optional):</strong>
            After reconstruction, you can fine-tune the visualization using the options below
            <details style="display:inline;">
                <summary style="display:inline;">(<strong>click to expand</strong>):</summary>
                <ul>
                <li><em>Confidence Threshold:</em> Adjust the filtering of points based on confidence.</li>
                <li><em>Show Points from Frame:</em> Select specific frames to display in the point cloud.</li>
                <li><em>Show Camera:</em> Toggle the display of estimated camera positions.</li>
                <li><em>Filter Sky / Filter Black Background:</em> Remove sky or black-background points.</li>
                <li><em>Select a Prediction Mode:</em> Choose between "Depthmap and Camera Branch" or "Pointmap Branch."</li>
                </ul>
            </details>
            </li>
        </ol>
        <p><strong style="color: #0ea5e9;">Please note:</strong> <span style="color: #0ea5e9; font-weight: bold;">VGGT typically reconstructs a scene in less than 1 second. However, visualizing 3D points may take tens of seconds due to third-party rendering, which are independent of VGGT's processing time. </span></p>
        </div>
        """
        )

        target_dir_output = gr.Textbox(label="Target Dir", visible=False, value="None")

        with gr.Row():
            with gr.Column(scale=2):
                input_video = gr.Video(label="Upload Video", interactive=True)
                input_images = gr.File(file_count="multiple", label="Upload Images", interactive=True)

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
                        "Please upload a video or images, then click Reconstruct.", elem_classes=["custom-log"]
                    )
                    reconstruction_output = gr.Model3D(height=520, zoom_speed=0.5, pan_speed=0.5)

                with gr.Row():
                    submit_btn = gr.Button("Reconstruct", scale=1, variant="primary")
                    clear_btn = gr.ClearButton(
                        [input_video, input_images, reconstruction_output, log_output, target_dir_output, image_gallery],
                        scale=1,
                    )

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
            [colosseum_video, "22", None, 20.0, False, False, True, False, "Depthmap and Camera Branch", "True"],
            [pyramid_video, "30", None, 35.0, False, False, True, False, "Depthmap and Camera Branch", "True"],
            [single_cartoon_video, "1", None, 15.0, False, False, True, False, "Depthmap and Camera Branch", "True"],
            [single_oil_painting_video, "1", None, 20.0, False, False, True, True, "Depthmap and Camera Branch", "True"],
            [room_video, "8", None, 5.0, False, False, True, False, "Depthmap and Camera Branch", "True"],
            [kitchen_video, "25", None, 50.0, False, False, True, False, "Depthmap and Camera Branch", "True"],
            [fern_video, "20", None, 45.0, False, False, True, False, "Depthmap and Camera Branch", "True"],
        ]

        def example_pipeline(
            input_video,
            num_images_str,
            input_images,
            conf_thres,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            is_example_str,
        ):
            """
            1) Copy example images to new target_dir
            2) Reconstruct
            3) Return model3D + logs + new_dir + updated dropdown + gallery
            We do NOT return is_example. It's just an input.
            """
            target_dir, image_paths = handle_uploads(input_video, input_images)
            # Always use "All" for frame_filter in examples
            frame_filter = "All"
            glbfile, log_msg, dropdown = gradio_demo(
                target_dir, conf_thres, frame_filter, mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode
            )
            return glbfile, log_msg, target_dir, dropdown, image_paths

        gr.Markdown("Click any row to load an example.", elem_classes=["example-log"])

        gr.Examples(
            examples=examples,
            inputs=[
                input_video,
                num_images,
                input_images,
                conf_thres,
                mask_black_bg,
                mask_white_bg,
                show_cam,
                mask_sky,
                prediction_mode,
                is_example,
            ],
            outputs=[reconstruction_output, log_output, target_dir_output, frame_filter, image_gallery],
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
        input_video.change(
            fn=update_gallery_on_upload,
            inputs=[input_video, input_images],
            outputs=[reconstruction_output, target_dir_output, image_gallery, log_output],
        )
        input_images.change(
            fn=update_gallery_on_upload,
            inputs=[input_video, input_images],
            outputs=[reconstruction_output, target_dir_output, image_gallery, log_output],
        )

        demo.queue(max_size=20).launch(show_error=True, share=False)

if __name__ == "__main__":
    main()