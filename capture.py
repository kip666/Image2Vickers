"""
Camera Video Feed Capture and Vickers Microhardness Indentation Analysis.
Provides live camera streaming, snapshot capture, automatic edge detection,
interactive point adjustment with real-time recalculation, Vickers hardness computation,
and multi-camera switching with persistent camera index storage in calibration.json.
"""

import datetime
import os
import sys
from pathlib import Path
import cv2

# Import analysis module
try:
    from indent_analyzer import (
        detect_indentation,
        annotate_image,
        interactive_measure_indent,
        interactive_calibrate_indent,
        prompt_target_hardness,
        calculate_vickers_hardness,
        calculate_diagonal_from_hardness,
        load_calibration,
        save_calibration,
        save_camera_index,
        DEFAULT_CALIBRATION_FILE
    )
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from indent_analyzer import (
        detect_indentation,
        annotate_image,
        interactive_measure_indent,
        interactive_calibrate_indent,
        prompt_target_hardness,
        calculate_vickers_hardness,
        calculate_diagonal_from_hardness,
        load_calibration,
        save_calibration,
        save_camera_index,
        DEFAULT_CALIBRATION_FILE
    )


def open_camera(cam_index, width=1280, height=720):
    """
    Opens camera by index using DirectShow on Windows, with fallback to default backend.
    Configures resolution.
    """
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_index)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def get_available_cameras(max_to_test=6):
    """
    Probes system for available camera device indices.
    """
    available = []
    for i in range(max_to_test):
        test_cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if test_cap.isOpened():
            available.append(i)
            test_cap.release()
    return available


def main():
    save_dir = Path(__file__).resolve().parent
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load calibration parameters
    calib = load_calibration(DEFAULT_CALIBRATION_FILE)
    scale_um = calib.get('scale_um_per_px', 0.164901) if calib else 0.164901
    load_gf = calib.get('default_load_gf', 300.0) if calib else 300.0

    # Load preferred camera index from calibration.json (fallback: 1 or sys.argv)
    saved_cam_idx = calib.get('camera_index', 1) if calib else 1
    cam_index = saved_cam_idx
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        cam_index = int(sys.argv[1])

    print("=" * 65)
    print("Microhardness Tester: Video Capture & Indent Analysis (Vickers)")
    print(f"Working directory: {save_dir}")
    print(f"Initial Camera:    Camera #{cam_index}")
    print(f"Test load:         {int(load_gf)} g ({load_gf/1000.0:.3f} kgf)")
    if scale_um:
        print(f"Scale:             {scale_um:.6f} um/pixel ({1.0/scale_um:.2f} px/um)")
    else:
        print("Scale calibration: NOT PERFORMED")
    print("-" * 65)
    print("Hotkeys:")
    print("  [SPACE] / [S] - Capture frame -> Interactive point adjustment -> Save")
    print("  [V] / [TAB]   - Switch between connected cameras (saved automatically)")
    print("  [M]           - Toggle live analysis in real time (ON/OFF)")
    print("  [C]           - Calibrate scale by entering test block hardness (HV)")
    print("  [Q] / [ESC]   - Exit")
    print("In calibration mode [C]:")
    print("  - Input certified microhardness (HV) from your test block")
    print("  - Align corner points with mouse to indent tips")
    print("  - Instant live scale calculation, press [SPACE/ENTER] to apply")
    print("In measurement window:")
    print("  - Drag diagonal corner points with mouse (in main view or zoom inset)")
    print("  - Real-time instant recalculation of D1, D2 and Vickers hardness")
    print("  - [R] - Reset to auto-detection")
    print("  - [ENTER/SPACE/ESC] - Finish and save to result file")
    print("=" * 65)

    cap = open_camera(cam_index)
    if not cap.isOpened():
        print(f"Warning: Camera #{cam_index} could not be opened. Scanning available cameras...")
        available = get_available_cameras()
        if available:
            cam_index = available[0]
            cap = open_camera(cam_index)
            save_camera_index(cam_index)
            print(f"Connected to Camera #{cam_index} (saved as preferred).")
        else:
            print("Error: No working cameras found on this computer.")
            return

    cam_win = "Microhardness Camera Feed"
    result_win = "Measurement Result - Vickers Hardness"
    cv2.namedWindow(cam_win, cv2.WINDOW_NORMAL)

    live_analysis_mode = False
    cam_switch_msg = None
    cam_switch_time = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Error reading camera frame.")
                break

            display_frame = frame.copy()
            h_disp, w_disp = display_frame.shape[:2]

            # Live recognition mode
            if live_analysis_mode:
                res = detect_indentation(frame)
                if res['success']:
                    display_frame = annotate_image(frame, res, scale_um, load_gf)
                else:
                    cv2.putText(
                        display_frame,
                        f"Searching for indent... [Camera #{cam_index}]",
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 165, 255),
                        2,
                        cv2.LINE_AA,
                    )
            else:
                scale_str = f"Scale: {scale_um:.4f} um/px" if scale_um else "No calib"
                cv2.putText(
                    display_frame,
                    f"CAM #{cam_index} | SPACE: Capture | V/TAB: Switch Cam | M: Live ({'ON' if live_analysis_mode else 'OFF'}) | C: Calib | Q: Exit  [{scale_str}]",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            # Show brief camera switch notification if active
            if cam_switch_msg and cam_switch_time:
                elapsed = (datetime.datetime.now() - cam_switch_time).total_seconds()
                if elapsed < 2.5:
                    t_box_sz, _ = cv2.getTextSize(cam_switch_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    bx1 = 20
                    by1 = h_disp - 50
                    bx2 = bx1 + t_box_sz[0] + 20
                    by2 = by1 + t_box_sz[1] + 16
                    cv2.rectangle(display_frame, (bx1, by1), (bx2, by2), (20, 20, 20), -1)
                    cv2.rectangle(display_frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                    cv2.putText(
                        display_frame,
                        cam_switch_msg,
                        (bx1 + 10, by2 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                else:
                    cam_switch_msg = None
                    cam_switch_time = None

            cv2.imshow(cam_win, display_frame)

            key = cv2.waitKey(1) & 0xFF

            # Exit
            if key in (ord('q'), ord('Q'), 27):
                print("Exiting application.")
                break

            # Switch connected cameras: [V] (Video), [TAB], or [K]
            if key in (ord('v'), ord('V'), 9, ord('k'), ord('K'), 236):
                avail = get_available_cameras(max_to_test=6)
                if not avail:
                    print("No cameras detected.")
                elif len(avail) == 1 and cam_index in avail:
                    print(f"Only one camera detected on system (Camera #{cam_index}).")
                    cam_switch_msg = f"Camera #{cam_index} (Only camera available)"
                    cam_switch_time = datetime.datetime.now()
                else:
                    if cam_index in avail:
                        idx_pos = avail.index(cam_index)
                        next_idx = avail[(idx_pos + 1) % len(avail)]
                    else:
                        next_idx = avail[0]

                    print(f"\nSwitching from Camera #{cam_index} to Camera #{next_idx}...")
                    cap.release()
                    new_cap = open_camera(next_idx)
                    if new_cap.isOpened():
                        cap = new_cap
                        cam_index = next_idx
                        save_camera_index(cam_index)
                        print(f"Successfully switched to Camera #{cam_index}. Saved to {DEFAULT_CALIBRATION_FILE.name}.")
                        cam_switch_msg = f"Switched to Camera #{cam_index} (Saved)"
                        cam_switch_time = datetime.datetime.now()
                    else:
                        print(f"Failed to open Camera #{next_idx}, reverting to Camera #{cam_index}...")
                        cap = open_camera(cam_index)

            # Toggle live analysis
            if key in (ord('m'), ord('M')):
                live_analysis_mode = not live_analysis_mode
                mode_str = "ENABLED" if live_analysis_mode else "DISABLED"
                print(f"Live analysis {mode_str}")

            # Scale calibration using certified reference microhardness (HV)
            if key in (ord('c'), ord('C')):
                print("\n" + "=" * 65)
                print("STARTING SCALE CALIBRATION (REFERENCE HARDNESS BLOCK)")
                print("Please enter the certified hardness (HV) of your calibration block.")

                # Default target HV from previous calibration
                prev_hv = calib.get('reference_hv') if calib else None
                if prev_hv is None:
                    prev_hv = 480.0

                target_hv = prompt_target_hardness(initial_val=prev_hv, load_gf=load_gf)
                if target_hv is None or target_hv <= 0:
                    print("Calibration cancelled.")
                    continue

                d_nom = calculate_diagonal_from_hardness(target_hv, load_gf)
                print(f"Target Hardness: {target_hv:.1f} HV ({int(load_gf)} gf) => Theoretical Diagonal: {d_nom:.2f} um")
                print("Opening calibration window. You can drag corner points to align with the indent.")
                print("Press [T] to edit hardness, [SPACE/ENTER] to save and apply, [ESC] to cancel.")

                res = detect_indentation(frame)
                calib_meas_path = save_dir / "calibration_latest_measured.jpg"

                new_scale, final_res, _ = interactive_calibrate_indent(
                    frame,
                    detection_result=res,
                    target_hv=target_hv,
                    load_gf=load_gf,
                    win_title="Vickers Calibration - Reference Hardness Mode",
                    save_path=calib_meas_path
                )

                if new_scale is not None and new_scale > 0:
                    scale_um = new_scale
                    calib = load_calibration(DEFAULT_CALIBRATION_FILE)
                    print("-" * 65)
                    print("CALIBRATION SUCCESSFUL AND APPLIED:")
                    print(f"  Reference Hardness: {target_hv:.1f} HV ({int(load_gf)} gf)")
                    print(f"  Nominal Diagonal:   {d_nom:.2f} um")
                    print(f"  Measured Diagonal:  {final_res['d_avg_px']:.2f} px")
                    print(f"  Calibrated Scale:   {scale_um:.6f} um/pixel ({1.0/scale_um:.2f} px/um)")
                    print(f"  Saved to: {DEFAULT_CALIBRATION_FILE.name}")
                    print(f"  Report saved to: {calib_meas_path.name}")
                    print("=" * 65)
                else:
                    print("Calibration cancelled without changes.")

            # Frame capture, interactive microhardness measurement and save on window close
            if key in (32, ord('s'), ord('S'), 13):
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
                raw_filename = f"frame_{timestamp}.jpg"
                raw_path = save_dir / raw_filename

                # 1. Save original clean frame
                cv2.imwrite(str(raw_path), frame)
                print(f"\n" + "=" * 65)
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Raw frame saved: {raw_filename}")

                # 2. Perform automated indent detection
                res = detect_indentation(frame)
                meas_filename = f"frame_{timestamp}_measured.jpg"
                meas_path = save_dir / meas_filename

                if res.get('success'):
                    print("  Indent detected automatically.")
                else:
                    err_msg = res.get('error', 'Unknown')
                    print(f"  Auto-detection: {err_msg}. You can place points manually.")

                print("  Measurement window opened:")
                print("    - Drag corner points TL, TR, BR, BL with mouse (in main view or zoom inset)")
                print("    - Diagonal lengths and Vickers hardness recalculate instantly")
                print("    - [R] - Reset to auto-detection")
                print("    - [SPACE / ENTER / ESC / Close window] - Finish and save result")

                # 3. Interactive measurement window: saving to meas_path occurs ONLY upon window close!
                final_res, _ = interactive_measure_indent(
                    frame,
                    detection_result=res,
                    scale_um_per_px=scale_um,
                    load_gf=load_gf,
                    win_title=result_win,
                    save_path=meas_path
                )

                if final_res and final_res.get('success'):
                    d1_px = final_res['d1_px']
                    d2_px = final_res['d2_px']
                    d_avg_px = final_res['d_avg_px']
                    d1_um = d1_px * scale_um
                    d2_um = d2_px * scale_um
                    d_avg_um = d_avg_px * scale_um
                    hv = calculate_vickers_hardness(d_avg_um, load_gf)

                    print("-" * 65)
                    print(f"  FINAL RESULTS (saved to {meas_filename}):")
                    print(f"  Camera Used:       Camera #{cam_index}")
                    print(f"  Load:              {int(load_gf)} g ({load_gf/1000.0:.3f} kgf)")
                    print(f"  Diagonal D1:      {d1_um:.2f} um  ({d1_px:.1f} px)")
                    print(f"  Diagonal D2:      {d2_um:.2f} um  ({d2_px:.1f} px)")
                    print(f"  Mean diagonal:    {d_avg_um:.2f} um  ({d_avg_px:.1f} px)")
                    print(f"  VICKERS HARDNESS: {hv:.1f} HV {load_gf/1000.0:.1f}")
                    if final_res.get('manual_adjusted'):
                        print(f"  Status:            Manually adjusted by user")
                    else:
                        print(f"  Status:            Automatic measurement")
                    print("=" * 65)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released, all windows closed.")


if __name__ == "__main__":
    main()
