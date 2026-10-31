import os
import cv2
import tempfile
import gradio as gr

from collections import Counter
from dotenv import load_dotenv
from inference_sdk import (
    InferenceHTTPClient,
    InferenceConfiguration
)


# =========================================================
# ROBOFLOW CONFIGURATION
# =========================================================

load_dotenv()

API_KEY = os.getenv("ROBOFLOW_API_KEY")

MODEL_ID = "waste-image-classification-298ab/15"

if not API_KEY:
    raise RuntimeError(
        "ROBOFLOW_API_KEY not found in .env file."
    )


# =========================================================
# CLASS COLORS
# =========================================================
# Define colors in HEX only.
# The same colors are used throughout the application.
# =========================================================

CLASS_HEX_COLORS = {

    "E_waste": "#8B00FF",

    "glass": "#FF8C00",

    "metal": "#FF1744",

    "mixed": "#00B8F0",

    "Organic": "#FF00CC",

    "Paper": "#9CFF00",

    "plastics": "#00E5B0",

    "Textile": "#FFF000",

}


# =========================================================
# HEX → OPENCV BGR
# =========================================================

def hex_to_bgr(hex_color):

    hex_color = hex_color.lstrip("#")

    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    return (b, g, r)


CLASS_COLORS = {

    class_name: hex_to_bgr(hex_color)

    for class_name, hex_color
    in CLASS_HEX_COLORS.items()

}


# =========================================================
# ROBOFLOW CLIENT
# =========================================================

client = InferenceHTTPClient(

    api_url="https://serverless.roboflow.com",

    api_key=API_KEY

).configure(

    InferenceConfiguration(

        api_key_transport="header",

        confidence_threshold=0.25,

        iou_threshold=0.5

    )

)


# =========================================================
# ROBOFLOW STYLE DETECTION DRAWING
# =========================================================

def draw_detections(image, predictions):

    output = image.copy()
    counts = Counter()

    image_height, image_width = output.shape[:2]

    # Roboflow-like thin, clean bounding boxes.
    # Keep line width small while scaling slightly for very large images.
    line_thickness = max(
        1,
        min(3, int(round(min(image_width, image_height) / 900)))
    )

    # Crisp but compact label typography.
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(
        0.48,
        min(0.85, min(image_width, image_height) / 900)
    )
    text_thickness = 1

    label_pad_x = 5
    label_pad_y = 4

    for prediction in predictions:

        x = float(prediction.get("x", 0))
        y = float(prediction.get("y", 0))
        box_width = float(prediction.get("width", 0))
        box_height = float(prediction.get("height", 0))
        confidence = float(prediction.get("confidence", 0))
        class_name = str(
            prediction.get("class", "unknown")
        )

        # Roboflow coordinates are center x/y + width/height.
        x1 = int(round(x - box_width / 2))
        y1 = int(round(y - box_height / 2))
        x2 = int(round(x + box_width / 2))
        y2 = int(round(y + box_height / 2))

        x1 = max(0, min(x1, image_width - 1))
        y1 = max(0, min(y1, image_height - 1))
        x2 = max(0, min(x2, image_width - 1))
        y2 = max(0, min(y2, image_height - 1))

        color = CLASS_COLORS.get(
            class_name,
            (255, 255, 255)
        )

        # -------------------------------------------------
        # THIN ROBoflow-STYLE OUTLINE
        # -------------------------------------------------
        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            color,
            line_thickness,
            cv2.LINE_AA
        )

        # -------------------------------------------------
        # LABEL
        # -------------------------------------------------
        label = f"{class_name} {confidence * 100:.0f}%"

        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            font,
            font_scale,
            text_thickness
        )

        label_width = (
            text_width
            + label_pad_x * 2
        )

        label_height = (
            text_height
            + baseline
            + label_pad_y * 2
        )

        # Place label directly above the box, like Roboflow.
        label_left = x1
        label_bottom = y1

        # If there isn't enough room above, put it inside the
        # upper edge of the bounding box.
        label_top = label_bottom - label_height

        if label_top < 0:
            label_top = y1
            label_bottom = min(
                image_height,
                y1 + label_height
            )

        label_right = min(
            image_width - 1,
            label_left + label_width
        )

        # Solid colored label background.
        cv2.rectangle(
            output,
            (label_left, label_top),
            (label_right, label_bottom),
            color,
            -1
        )

        # Dark/black text gives maximum contrast on bright label colors.
        # This resembles the compact Roboflow visualization.
        text_color = (0, 0, 0)

        text_x = (
            label_left
            + label_pad_x
        )

        text_y = (
            label_bottom
            - label_pad_y
            - baseline
        )

        cv2.putText(
            output,
            label,
            (text_x, text_y),
            font,
            font_scale,
            text_color,
            text_thickness,
            cv2.LINE_AA
        )

        counts[class_name] += 1

    return output, counts


# =========================================================
# IMAGE DETECTION
# =========================================================

def detect_image(
    image_path,
    confidence
):

    if image_path is None:
        return (
            None,
            """
            <div class="report-card">
                <div class="report-status neutral">
                    ● WAITING FOR IMAGE
                </div>
                <div class="report-title">
                    No Image Selected
                </div>
                <div class="report-text">
                    Please upload an image to begin detection.
                </div>
            </div>
            """
        )

    try:

        client.configure(
            InferenceConfiguration(
                api_key_transport="header",
                confidence_threshold=confidence,
                iou_threshold=0.5
            )
        )

        result = client.infer(
            image_path,
            model_id=MODEL_ID
        )

        predictions = result.get(
            "predictions",
            []
        )

        # Read the ORIGINAL image at its full resolution.
        image = cv2.imread(
            image_path,
            cv2.IMREAD_COLOR
        )

        if image is None:
            return (
                None,
                """
                <div class="report-card error-card">
                    <div class="report-status error">
                        ● IMAGE ERROR
                    </div>
                    <div class="report-title">
                        Could Not Read Image
                    </div>
                </div>
                """
            )

        # Draw detections on the original-resolution image.
        output, counts = draw_detections(
            image,
            predictions
        )

        # Save as PNG to avoid JPEG compression and preserve clarity.
        output_dir = tempfile.mkdtemp(
            prefix="waste_detection_"
        )

        output_path = os.path.join(
            output_dir,
            "detection_result.png"
        )

        cv2.imwrite(
            output_path,
            output,
            [
                cv2.IMWRITE_PNG_COMPRESSION,
                0
            ]
        )

        # No detections.
        if not predictions:
            return (
                output_path,
                """
                <div class="report-card">
                    <div class="report-status neutral">
                        ● ANALYSIS COMPLETE
                    </div>
                    <div class="report-title">
                        No Waste Objects Detected
                    </div>
                    <div class="report-text">
                        No objects were detected above
                        the selected confidence threshold.
                    </div>
                </div>
                """
            )

        total_objects = len(
            predictions
        )

        report = f"""
        <div class="report-card">

            <div class="report-top">

                <div>

                    <div class="report-status success">
                        ● ANALYSIS COMPLETE
                    </div>

                    <div class="report-title">
                        Detection Summary
                    </div>

                </div>

                <div class="total-box">

                    <div class="total-number">
                        {total_objects}
                    </div>

                    <div class="total-label">
                        OBJECTS
                    </div>

                </div>

            </div>

            <div class="report-divider"></div>

            <div class="classes-title">
                🏷️ Detected Categories
            </div>
        """

        for class_name, count in counts.items():

            css_color = CLASS_HEX_COLORS.get(
                class_name,
                "#FFFFFF"
            )

            report += f"""
            <div class="detected-row">

                <div class="detected-left">

                    <span
                        class="class-dot"
                        style="
                            background:{css_color};
                            box-shadow:
                            0 0 10px {css_color};
                        "
                    ></span>

                    <span>
                        {class_name}
                    </span>

                </div>

                <div class="detected-count">
                    {count}
                </div>

            </div>
            """

        report += """
        </div>
        """

        return (
            output_path,
            report
        )

    except Exception as e:

        print(
            "\nROBOFLOW IMAGE ERROR:"
        )

        print(
            repr(e)
        )

        return (
            None,
            f"""
            <div class="report-card error-card">

                <div class="report-status error">
                    ● DETECTION ERROR
                </div>

                <div class="report-title">
                    Roboflow Error
                </div>

                <div class="error-message">
                    {str(e)}
                </div>

            </div>
            """
        )


# =========================================================
# VIDEO DETECTION
# =========================================================

def detect_video(
    video_path,
    confidence,
    frame_skip
):

    if video_path is None:

        return (

            None,

            """

            <div class="report-card">

                <div class="report-status neutral">
                    ● WAITING FOR VIDEO
                </div>

                <div class="report-title">
                    No Video Selected
                </div>

            </div>

            """

        )


    cap = cv2.VideoCapture(

        video_path

    )


    if not cap.isOpened():

        return (

            None,

            """

            <div class="report-card error-card">

                <div class="report-status error">
                    ● VIDEO ERROR
                </div>

                <div class="report-title">
                    Could Not Open Video
                </div>

            </div>

            """

        )


    fps = cap.get(

        cv2.CAP_PROP_FPS

    )


    if fps <= 0:

        fps = 25


    width = int(

        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )

    )


    height = int(

        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )

    )


    output_dir = tempfile.mkdtemp()


    raw_video = os.path.join(

        output_dir,

        "waste_detection.mp4"

    )


    final_video = os.path.join(

        output_dir,

        "waste_detection_final.mp4"

    )


    fourcc = cv2.VideoWriter_fourcc(

        *"mp4v"

    )


    writer = cv2.VideoWriter(

        raw_video,

        fourcc,

        fps,

        (width, height)

    )


    frame_number = 0

    total_detections = 0

    class_counts = Counter()

    last_predictions = []


    try:

        client.configure(

            InferenceConfiguration(

                api_key_transport="header",

                confidence_threshold=confidence,

                iou_threshold=0.5

            )

        )


        while True:

            ret, frame = cap.read()


            if not ret:

                break


            frame_number += 1


            # =============================================
            # INFERENCE EVERY N FRAMES
            # =============================================

            if frame_number % int(
                frame_skip
            ) == 0:

                rgb_frame = cv2.cvtColor(

                    frame,

                    cv2.COLOR_BGR2RGB

                )


                result = client.infer(

                    rgb_frame,

                    model_id=MODEL_ID

                )


                last_predictions = result.get(

                    "predictions",

                    []

                )


                total_detections += len(

                    last_predictions

                )


                for prediction in last_predictions:

                    class_name = prediction.get(

                        "class",

                        "unknown"

                    )


                    class_counts[

                        class_name

                    ] += 1


            # =============================================
            # DRAW
            # =============================================

            frame, _ = draw_detections(

                frame,

                last_predictions

            )


            writer.write(

                frame

            )


    except Exception as e:

        print(
            "\nROBOFLOW VIDEO ERROR:"
        )

        print(
            repr(e)
        )


        cap.release()

        writer.release()


        return (

            None,

            f"""

            <div class="report-card error-card">

                <div class="report-status error">
                    ● VIDEO ERROR
                </div>

                <div class="report-title">
                    Roboflow Video Error
                </div>

                <div class="error-message">
                    {str(e)}
                </div>

            </div>

            """

        )


    cap.release()

    writer.release()


    # =====================================================
    # H.264 CONVERSION
    # =====================================================

    ffmpeg_command = (

        f'ffmpeg -y '

        f'-i "{raw_video}" '

        f'-c:v libx264 '

        f'-pix_fmt yuv420p '

        f'"{final_video}"'

    )


    exit_code = os.system(

        ffmpeg_command
        + " >nul 2>&1"

    )


    if (

        exit_code == 0

        and

        os.path.exists(
            final_video
        )

    ):

        video_result = final_video

    else:

        video_result = raw_video


    # =====================================================
    # REPORT
    # =====================================================

    report = f"""

    <div class="report-card">

        <div class="report-top">

            <div>

                <div class="report-status success">
                    ● VIDEO ANALYSIS COMPLETE
                </div>

                <div class="report-title">
                    Video Detection Summary
                </div>

            </div>

            <div class="total-box">

                <div class="total-number">
                    {total_detections}
                </div>

                <div class="total-label">
                    DETECTIONS
                </div>

            </div>

        </div>


        <div class="report-divider"></div>


        <div class="video-stats">

            <div class="video-stat">

                <div class="stat-value">
                    {frame_number}
                </div>

                <div class="stat-label">
                    PROCESSED FRAMES
                </div>

            </div>


            <div class="video-stat">

                <div class="stat-value">
                    {frame_skip}
                </div>

                <div class="stat-label">
                    FRAME SKIP
                </div>

            </div>


            <div class="video-stat">

                <div class="stat-value">
                    {confidence:.2f}
                </div>

                <div class="stat-label">
                    CONFIDENCE
                </div>

            </div>

        </div>


        <div class="classes-title">
            🏷️ Detected Categories
        </div>

    """


    if class_counts:

        for class_name, count in class_counts.items():

            color = CLASS_HEX_COLORS.get(

                class_name,

                "#FFFFFF"

            )


            report += f"""

            <div class="detected-row">

                <div class="detected-left">

                    <span
                        class="class-dot"
                        style="
                            background:{color};
                            box-shadow:
                            0 0 10px {color};
                        "
                    ></span>

                    <span>
                        {class_name}
                    </span>

                </div>

                <div class="detected-count">
                    {count}
                </div>

            </div>

            """


    else:

        report += """

        <div class="report-text">

            No waste objects were detected.

        </div>

        """


    report += """

    </div>

    """


    return (

        video_result,

        report

    )


# =========================================================
# PREMIUM CSS
# =========================================================

custom_css = """

/* =====================================================
   GLOBAL
   ===================================================== */

:root {

    --bg-main: #07111f;

    --bg-card: rgba(15, 30, 48, 0.88);

    --border:
        rgba(148,163,184,0.15);

    --primary: #22c55e;

    --cyan: #06b6d4;

}


body {

    margin: 0 !important;

    background:

        radial-gradient(
            circle at 8% 8%,
            rgba(34,197,94,0.10),
            transparent 30%
        ),

        radial-gradient(
            circle at 92% 15%,
            rgba(6,182,212,0.09),
            transparent 30%
        ),

        #07111f !important;

}


.gradio-container {

    max-width: 1500px !important;

    margin: auto !important;

    background: transparent !important;

    color: #f8fafc !important;

    padding:
        25px
        35px
        55px
        35px !important;

}


/* =====================================================
   HERO
   ===================================================== */

.hero {

    position: relative;

    overflow: hidden;

    padding:
        42px
        45px;

    border-radius:
        28px;

    background:

        linear-gradient(
            135deg,
            rgba(17,42,60,0.97),
            rgba(7,25,40,0.97)
        );

    border:
        1px solid
        rgba(148,163,184,0.18);

    box-shadow:

        0 25px 70px
        rgba(0,0,0,0.35);

    margin-bottom:
        28px;

}


.hero:before {

    content: "";

    position: absolute;

    width: 350px;

    height: 350px;

    right: -130px;

    top: -190px;

    border-radius: 50%;

    background:
        rgba(34,197,94,0.14);

    filter:
        blur(15px);

}


.hero:after {

    content: "";

    position: absolute;

    width: 250px;

    height: 250px;

    left: 40%;

    bottom: -190px;

    border-radius: 50%;

    background:
        rgba(6,182,212,0.10);

    filter:
        blur(20px);

}


.hero-content {

    position:
        relative;

    z-index: 2;

}


.hero-title {

    font-size:
        44px;

    font-weight:
        850;

    letter-spacing:
        -1.7px;

    color:
        white;

}


.hero-title span {

    color:
        #4ade80;

}


.hero-subtitle {

    color:
        #a7b7c9;

    font-size:
        16px;

    max-width:
        760px;

    line-height:
        1.75;

}


/* =====================================================
   STATUS
   ===================================================== */

.status-row {

    display:
        flex;

    gap:
        10px;

    margin-top:
        23px;

    flex-wrap:
        wrap;

}


.status-badge {

    display:
        inline-flex;

    align-items:
        center;

    gap:
        8px;

    padding:
        8px 15px;

    border-radius:
        50px;

    background:
        rgba(255,255,255,0.055);

    border:
        1px solid
        rgba(255,255,255,0.10);

    color:
        #cbd5e1;

    font-size:
        13px;

}


/* =====================================================
   GLASS CARD
   ===================================================== */

.glass-card {

    background:

        linear-gradient(
            145deg,
            rgba(18,36,56,0.92),
            rgba(9,23,39,0.92)
        ) !important;

    border:
        1px solid
        rgba(148,163,184,0.13) !important;

    border-radius:
        22px !important;

    padding:
        23px !important;

    box-shadow:

        0 15px 45px
        rgba(0,0,0,0.22);

    backdrop-filter:
        blur(18px);

    transition:
        all 0.25s ease;

}


.glass-card:hover {

    border-color:
        rgba(74,222,128,0.25) !important;

    box-shadow:

        0 20px 55px
        rgba(0,0,0,0.30);

}


/* =====================================================
   SECTION
   ===================================================== */

.section-title {

    font-size:
        21px;

    font-weight:
        750;

    color:
        #f8fafc;

    margin:
        25px 0
        7px 3px;

}


.section-description {

    color:
        #8292a7;

    font-size:
        13px;

    margin-bottom:
        17px;

}


/* =====================================================
   CLASS GRID
   ===================================================== */

.class-grid {

    display:
        grid;

    grid-template-columns:
        repeat(4, minmax(0,1fr));

    gap:
        11px;

}


.class-item {

    display:
        flex;

    align-items:
        center;

    gap:
        11px;

    padding:
        14px;

    border-radius:
        14px;

    background:
        rgba(255,255,255,0.035);

    border:
        1px solid
        rgba(148,163,184,0.09);

    transition:
        all 0.2s ease;

}


.class-item:hover {

    background:
        rgba(255,255,255,0.075);

    transform:
        translateY(-2px);

}


.class-color {

    width:
        11px;

    height:
        11px;

    min-width:
        11px;

    border-radius:
        50%;

    box-shadow:
        0 0 10px currentColor;

}


.class-name {

    color:
        #dbeafe;

    font-size:
        13px;

    font-weight:
        600;

}


/* =====================================================
   INPUT
   ===================================================== */

.input-title {

    color:
        #e2e8f0;

    font-size:
        15px;

    font-weight:
        700;

    margin-bottom:
        12px;

}


.upload-area {

    border-radius:
        17px !important;

    border:
        1px dashed
        rgba(148,163,184,0.30) !important;

    background:
        rgba(255,255,255,0.025) !important;

}


/* =====================================================
   BUTTON
   ===================================================== */

.detect-btn {

    min-height:
        55px !important;

    border-radius:
        15px !important;

    border:
        none !important;

    background:

        linear-gradient(
            135deg,
            #22c55e,
            #16a34a
        ) !important;

    color:
        white !important;

    font-size:
        15px !important;

    font-weight:
        800 !important;

    box-shadow:

        0 8px 25px
        rgba(34,197,94,0.22);

    transition:
        all 0.2s ease;

}


.detect-btn:hover {

    transform:
        translateY(-2px);

    box-shadow:

        0 12px 35px
        rgba(34,197,94,0.35);

}


/* =====================================================
   REPORT
   ===================================================== */

.report-card {

    margin-top:
        15px;

    padding:
        20px;

    border-radius:
        17px;

    background:
        rgba(255,255,255,0.035);

    border:
        1px solid
        rgba(148,163,184,0.10);

    color:
        #cbd5e1;

}


.report-top {

    display:
        flex;

    justify-content:
        space-between;

    align-items:
        center;

    gap:
        20px;

}


.report-status {

    font-size:
        10px;

    font-weight:
        800;

    letter-spacing:
        1px;

    margin-bottom:
        6px;

}


.report-status.success {

    color:
        #4ade80;

}


.report-status.neutral {

    color:
        #94a3b8;

}


.report-status.error {

    color:
        #fb7185;

}


.report-title {

    font-size:
        19px;

    font-weight:
        750;

    color:
        #f1f5f9;

}


.report-text {

    color:
        #8191a6;

    font-size:
        13px;

    margin-top:
        5px;

}


.total-box {

    text-align:
        center;

    min-width:
        70px;

    padding:
        10px 14px;

    border-radius:
        14px;

    background:
        rgba(34,197,94,0.07);

    border:
        1px solid
        rgba(34,197,94,0.13);

}


.total-number {

    color:
        #4ade80;

    font-size:
        28px;

    font-weight:
        850;

}


.total-label {

    color:
        #64748b;

    font-size:
        9px;

    font-weight:
        700;

    letter-spacing:
        1px;

}


.report-divider {

    height:
        1px;

    background:
        rgba(148,163,184,0.10);

    margin:
        18px 0;

}


.classes-title {

    color:
        #cbd5e1;

    font-size:
        13px;

    font-weight:
        700;

    margin-bottom:
        11px;

}


.detected-row {

    display:
        flex;

    align-items:
        center;

    justify-content:
        space-between;

    padding:
        10px 12px;

    border-radius:
        10px;

    background:
        rgba(255,255,255,0.025);

    margin-bottom:
        6px;

}


.detected-left {

    display:
        flex;

    align-items:
        center;

    gap:
        10px;

    color:
        #dbeafe;

    font-size:
        13px;

    font-weight:
        600;

}


.class-dot {

    width:
        9px;

    height:
        9px;

    border-radius:
        50%;

}


.detected-count {

    color:
        #f8fafc;

    font-size:
        13px;

    font-weight:
        800;

}


/* =====================================================
   LIVE CAMERA SOURCE / OUTPUT
   ===================================================== */

.camera-source {
    margin-bottom: 10px !important;
}

.live-output {
    border-radius: 16px !important;
    overflow: hidden !important;
    background: #020817 !important;
}

.live-output img {
    object-fit: contain !important;
}

/* =====================================================
   LIVE HEADER
   ===================================================== */

.live-header {

    display:
        flex;

    align-items:
        center;

    justify-content:
        space-between;

    padding:
        20px 22px;

    margin-bottom:
        18px;

    border-radius:
        18px;

    background:

        linear-gradient(
            135deg,
            rgba(17,42,60,0.92),
            rgba(8,25,40,0.92)
        );

    border:
        1px solid
        rgba(148,163,184,0.14);

}


.live-title {

    color:
        #f8fafc;

    font-size:
        23px;

    font-weight:
        800;

}


.live-description {

    color:
        #8191a6;

    font-size:
        13px;

    margin-top:
        5px;

}


/* =====================================================
   LIVE DASHBOARD
   ===================================================== */

.live-dashboard {

    padding:
        18px;

    border-radius:
        17px;

    background:
        rgba(255,255,255,0.025);

    border:
        1px solid
        rgba(148,163,184,0.10);

}


.live-dashboard-header {

    display:
        flex;

    align-items:
        center;

    justify-content:
        space-between;

    gap:
        10px;

    margin-bottom:
        17px;

}


.live-dashboard-title {

    color:
        #e2e8f0;

    font-size:
        15px;

    font-weight:
        750;

}


.live-dashboard-subtitle {

    color:
        #64748b;

    font-size:
        11px;

    margin-top:
        3px;

}


.live-active,
.live-stopped {

    display:
        flex;

    align-items:
        center;

    gap:
        7px;

    padding:
        7px 12px;

    border-radius:
        50px;

    font-size:
        10px;

    font-weight:
        800;

    letter-spacing:
        0.7px;

}


.live-active {

    color:
        #4ade80;

    background:
        rgba(34,197,94,0.08);

    border:
        1px solid
        rgba(34,197,94,0.18);

}


.live-stopped {

    color:
        #94a3b8;

    background:
        rgba(148,163,184,0.07);

    border:
        1px solid
        rgba(148,163,184,0.12);

}


.status-live-dot,
.status-off-dot {

    width:
        7px;

    height:
        7px;

    border-radius:
        50%;

}


.status-live-dot {

    background:
        #22c55e;

    box-shadow:
        0 0 10px
        #22c55e;

    animation:
        livePulse 1.2s infinite;

}


.status-off-dot {

    background:
        #64748b;

}


/* =====================================================
   LIVE STAT CARDS
   ===================================================== */

.live-stat-grid {

    display:
        grid;

    grid-template-columns:
        repeat(3,1fr);

    gap:
        8px;

    margin-bottom:
        18px;

}


.live-stat-card {

    padding:
        13px 8px;

    text-align:
        center;

    border-radius:
        12px;

    background:
        rgba(255,255,255,0.035);

    border:
        1px solid
        rgba(148,163,184,0.08);

}


.live-stat-icon {

    font-size:
        13px;

    margin-bottom:
        4px;

}


.live-stat-value {

    color:
        #4ade80;

    font-size:
        20px;

    font-weight:
        850;

}


.live-stat-label {

    color:
        #64748b;

    font-size:
        8px;

    font-weight:
        750;

    letter-spacing:
        0.7px;

    margin-top:
        3px;

}


/* =====================================================
   LIVE CLASSES
   ===================================================== */

.live-category-title {

    color:
        #cbd5e1;

    font-size:
        12px;

    font-weight:
        750;

    margin-bottom:
        9px;

}


.live-detection-class {

    display:
        flex;

    align-items:
        center;

    justify-content:
        space-between;

    padding:
        9px 10px;

    margin-bottom:
        5px;

    border-radius:
        9px;

    background:
        rgba(255,255,255,0.025);

    color:
        #cbd5e1;

    font-size:
        11px;

}


.live-detection-class strong {

    color:
        #f8fafc;

}


.live-detection-class > span:nth-child(2) {

    flex:
        1;

    margin-left:
        8px;

}


.live-color-dot {

    width:
        8px;

    height:
        8px;

    border-radius:
        50%;

}


.no-live-detection {

    padding:
        12px;

    text-align:
        center;

    color:
        #64748b;

    font-size:
        11px;

}


.live-error {

    margin-top:
        10px;

    padding:
        12px;

    border-radius:
        10px;

    background:
        rgba(251,113,133,0.07);

    border:
        1px solid
        rgba(251,113,133,0.15);

    color:
        #fda4af;

    font-size:
        11px;

    word-break:
        break-word;

}


/* =====================================================
   LIVE PULSE
   ===================================================== */

@keyframes livePulse {

    0% {

        opacity:
            1;

        transform:
            scale(1);

    }

    50% {

        opacity:
            0.4;

        transform:
            scale(0.75);

    }

    100% {

        opacity:
            1;

        transform:
            scale(1);

    }

}


/* =====================================================
   CLEAR DETECTION IMAGE
   ===================================================== */

.image-result-clear img {
    image-rendering: auto !important;
    object-fit: contain !important;
}

.image-result-clear .wrap {
    min-height: 650px !important;
    background: #0b1220 !important;
}

/* =====================================================
   VIDEO STATS
   ===================================================== */

.video-stats {

    display:
        grid;

    grid-template-columns:
        repeat(3,1fr);

    gap:
        9px;

    margin-bottom:
        20px;

}


.video-stat {

    text-align:
        center;

    padding:
        12px;

    border-radius:
        12px;

    background:
        rgba(255,255,255,0.025);

}


.stat-value {

    color:
        #4ade80;

    font-size:
        19px;

    font-weight:
        800;

}


.stat-label {

    color:
        #64748b;

    font-size:
        9px;

    font-weight:
        700;

    letter-spacing:
        0.6px;

}


/* =====================================================
   TABS
   ===================================================== */

.tabs {

    margin-top:
        27px !important;

}


.tab-nav {

    gap:
        8px !important;

    border-bottom:
        1px solid
        rgba(148,163,184,0.12) !important;

    margin-bottom:
        20px !important;

}


.tab-nav button {

    border-radius:
        12px 12px 0 0 !important;

    padding:
        13px 22px !important;

    color:
        #8fa1b5 !important;

    font-weight:
        650 !important;

}


.tab-nav button:hover {

    color:
        #ffffff !important;

    background:
        rgba(255,255,255,0.04) !important;

}


/* =====================================================
   FOOTER
   ===================================================== */

.footer {

    text-align:
        center;

    margin-top:
        38px;

    padding-top:
        22px;

    border-top:
        1px solid
        rgba(148,163,184,0.10);

    color:
        #64748b;

    font-size:
        12px;

}


/* =====================================================
   RESPONSIVE
   ===================================================== */

@media (max-width: 1000px) {

    .class-grid {

        grid-template-columns:
            repeat(2,1fr);

    }

}


@media (max-width: 700px) {

    .gradio-container {

        padding:
            15px !important;

    }

    .hero {

        padding:
            28px 22px;

    }

    .hero-title {

        font-size:
            30px;

    }

    .live-header {

        flex-direction:
            column;

        align-items:
            flex-start;

    }

}


@media (max-width: 500px) {

    .class-grid {

        grid-template-columns:
            1fr;

    }

    .live-stat-grid {

        grid-template-columns:
            1fr;

    }

    .video-stats {

        grid-template-columns:
            1fr;

    }

}

"""


# =========================================================
# GRADIO APPLICATION
# =========================================================

with gr.Blocks(

    title="Smart Waste Detection System",

    css=custom_css,

    theme=gr.themes.Base(

        primary_hue="green",

        secondary_hue="cyan",

        neutral_hue="slate"

    )

) as demo:


    # =====================================================
    # HERO
    # =====================================================

    gr.HTML(
        """
        <div class="hero">

            <div class="hero-content">

                <div class="hero-title">

                    ♻️ Smart
                    <span>Waste Detection</span>

                </div>

                <div class="hero-subtitle">

                    AI-powered waste detection and
                    classification using YOLO11 and
                    Roboflow. Analyze waste through
                    images or videos.

                </div>

                <div class="status-row">

                    <div class="status-badge">

                        <span class="status-dot"></span>

                        AI System Online

                    </div>

                    <div class="status-badge">
                        ⚡ YOLO11
                    </div>

                    <div class="status-badge">
                        ☁️ Roboflow
                    </div>

                    <div class="status-badge">
                        ♻️ 8 Waste Classes
                    </div>

                </div>

            </div>

        </div>
        """
    )


    # =====================================================
    # WASTE CLASSIFICATION
    # =====================================================

    gr.HTML(
        """
        <div class="section-title">

            🎨 Waste Classification

        </div>

        <div class="section-description">

            Each waste category uses a unique color
            for its detection bounding box.

        </div>

        <div class="glass-card">

            <div class="class-grid">


                <div class="class-item">

                    <span
                        class="class-color"
                        style="
                            background:#8B00FF;
                            color:#8B00FF;
                        "
                    ></span>

                    <span class="class-name">
                        E_waste
                    </span>

                </div>


                <div class="class-item">

                    <span
                        class="class-color"
                        style="
                            background:#FF8C00;
                            color:#FF8C00;
                        "
                    ></span>

                    <span class="class-name">
                        Glass
                    </span>

                </div>


                <div class="class-item">

                    <span
                        class="class-color"
                        style="
                            background:#FF1744;
                            color:#FF1744;
                        "
                    ></span>

                    <span class="class-name">
                        Metal
                    </span>

                </div>


                <div class="class-item">

                    <span
                        class="class-color"
                        style="
                            background:#00B8F0;
                            color:#00B8F0;
                        "
                    ></span>

                    <span class="class-name">
                        Mixed
                    </span>

                </div>


                <div class="class-item">

                    <span
                        class="class-color"
                        style="
                            background:#FF00CC;
                            color:#FF00CC;
                        "
                    ></span>

                    <span class="class-name">
                        Organic
                    </span>

                </div>


                <div class="class-item">

                    <span
                        class="class-color"
                        style="
                            background:#9CFF00;
                            color:#9CFF00;
                        "
                    ></span>

                    <span class="class-name">
                        Paper
                    </span>

                </div>


                <div class="class-item">

                    <span
                        class="class-color"
                        style="
                            background:#00E5B0;
                            color:#00E5B0;
                        "
                    ></span>

                    <span class="class-name">
                        Plastics
                    </span>

                </div>


                <div class="class-item">

                    <span
                        class="class-color"
                        style="
                            background:#FFF000;
                            color:#FFF000;
                        "
                    ></span>

                    <span class="class-name">
                        Textile
                    </span>

                </div>


            </div>

        </div>
        """
    )


    # =====================================================
    # TABS
    # =====================================================

    with gr.Tabs(
        elem_classes="tabs"
    ):


        # =================================================
        # IMAGE DETECTION
        # =================================================

        with gr.Tab(
            "🖼️  Image Detection"
        ):

            with gr.Row(
                equal_height=False
            ):


                # =========================================
                # INPUT
                # =========================================

                with gr.Column(
                    scale=1,
                    elem_classes="glass-card"
                ):

                    gr.HTML(
                        """
                        <div class="input-title">

                            📤 Upload Image

                        </div>
                        """
                    )


                    image_input = gr.Image(

                        type="filepath",

                        label="",

                        height=430,

                        elem_classes="upload-area"

                    )


                    gr.Markdown(
                        "### ⚙️ Detection Settings"
                    )


                    image_confidence = gr.Slider(

                        minimum=0.05,

                        maximum=0.95,

                        value=0.25,

                        step=0.05,

                        label="Confidence Threshold"

                    )


                    gr.Markdown(
                        """
                        <div style="
                            color:#64748b;
                            font-size:11px;
                            line-height:1.7;
                            margin-bottom:15px;
                        ">

                        Higher confidence values produce
                        fewer but more confident detections.

                        </div>
                        """
                    )


                    image_button = gr.Button(

                        "🔍  ANALYZE WASTE",

                        variant="primary",

                        elem_classes="detect-btn"

                    )


                # =========================================
                # RESULT
                # =========================================

                with gr.Column(
                    scale=2,
                    elem_classes="glass-card"
                ):

                    gr.HTML(
                        """
                        <div class="input-title">

                            🎯 Detection Results

                        </div>
                        """
                    )


                    image_output = gr.Image(

                        type="filepath",

                        label="",

                        height=620,

                        elem_classes="image-result-clear"

                    )


                    image_report = gr.HTML(

                        """
                        <div class="report-card">

                            <div class="report-status neutral">
                                ● SYSTEM READY
                            </div>

                            <div class="report-title">
                                Ready for Analysis
                            </div>

                            <div class="report-text">

                                Upload an image and click
                                <b>Analyze Waste</b>.

                            </div>

                        </div>
                        """

                    )


            image_button.click(

                fn=detect_image,

                inputs=[

                    image_input,

                    image_confidence

                ],

                outputs=[

                    image_output,

                    image_report

                ]

            )


        # =================================================
        # VIDEO DETECTION
        # =================================================

        with gr.Tab(
            "🎥  Video Detection"
        ):

            with gr.Row(
                equal_height=False
            ):


                # =========================================
                # VIDEO INPUT
                # =========================================

                with gr.Column(
                    scale=1,
                    elem_classes="glass-card"
                ):

                    gr.HTML(
                        """
                        <div class="input-title">

                            🎬 Upload Video

                        </div>
                        """
                    )


                    video_input = gr.Video(

                        label="",

                        height=330

                    )


                    gr.Markdown(
                        "### ⚙️ Video Settings"
                    )


                    video_confidence = gr.Slider(

                        minimum=0.05,

                        maximum=0.95,

                        value=0.25,

                        step=0.05,

                        label="Confidence Threshold"

                    )


                    frame_skip = gr.Slider(

                        minimum=1,

                        maximum=5,

                        value=2,

                        step=1,

                        label="Frame Skip"

                    )


                    video_button = gr.Button(

                        "🎬  ANALYZE VIDEO",

                        variant="primary",

                        elem_classes="detect-btn"

                    )


                # =========================================
                # VIDEO RESULT
                # =========================================

                with gr.Column(
                    scale=2,
                    elem_classes="glass-card"
                ):

                    gr.HTML(
                        """
                        <div class="input-title">

                            🎯 Processed Video

                        </div>
                        """
                    )


                    video_output = gr.Video(

                        label="",

                        height=480

                    )


                    video_report = gr.HTML(

                        """
                        <div class="report-card">

                            <div class="report-status neutral">
                                ● SYSTEM READY
                            </div>

                            <div class="report-title">
                                Video Analysis Ready
                            </div>

                            <div class="report-text">

                                Upload a video and start
                                AI-powered detection.

                            </div>

                        </div>
                        """

                    )


            video_button.click(

                fn=detect_video,

                inputs=[

                    video_input,

                    video_confidence,

                    frame_skip

                ],

                outputs=[

                    video_output,

                    video_report

                ]

            )


    # =====================================================
    # FOOTER
    # =====================================================

    gr.HTML(
        """
        <div class="footer">

            ♻️ <b>Smart Waste Detection System</b>

            &nbsp; • &nbsp;

            YOLO11 + Roboflow

            &nbsp; • &nbsp;

            AI-Powered Waste Classification

            <br><br>

            <span style="color:#475569">

                Intelligent
                &nbsp;•&nbsp;
                Fast
                &nbsp;•&nbsp;
                Sustainable

            </span>

        </div>
        """
    )


# =========================================================
# START APPLICATION
# =========================================================

if __name__ == "__main__":

    demo.launch(

        server_name="127.0.0.1",
        share=False,

        inbrowser=True

    )