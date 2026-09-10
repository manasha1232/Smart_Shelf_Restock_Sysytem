"""
Smart Shelf Restock Alert System
================================
An AI-powered computer vision system for real-time multi-shelf inventory monitoring, 
missing item detection, and automated stock restoration alerting using YOLO and OpenCV.

Author: Antigravity AI
License: MIT
"""

import argparse
import json
import os
import sys
from datetime import datetime
import cv2
import numpy as np
from ultralytics import YOLO


class ShelfZone:
    """Represents a specific shelf region of interest (ROI) and its expected inventory."""

    def __init__(self, zone_id, name, y_min_pct, y_max_pct, expected_inventory):
        """
        :param zone_id: Unique integer ID for the shelf zone
        :param name: Human-readable name (e.g., 'Top Shelf')
        :param y_min_pct: Upper boundary percentage of image height (0.0 to 1.0)
        :param y_max_pct: Lower boundary percentage of image height (0.0 to 1.0)
        :param expected_inventory: Dict of item name -> expected count, e.g. {"cup": 2, "book": 5}
        """
        self.zone_id = zone_id
        self.name = name
        self.y_min_pct = y_min_pct
        self.y_max_pct = y_max_pct
        self.expected_inventory = expected_inventory

    def contains_point(self, cy, image_height):
        """Check if an object center y-coordinate falls within this shelf zone."""
        y_min = int(self.y_min_pct * image_height)
        y_max = int(self.y_max_pct * image_height)
        return y_min <= cy < y_max


class SmartShelfMonitor:
    """Core engine for detecting objects, assigning them to shelves, and generating restock alerts."""

    DEFAULT_SHELVES = [
        ShelfZone(
            zone_id=1,
            name="Shelf 1 (Top)",
            y_min_pct=0.0,
            y_max_pct=0.33,
            expected_inventory={"cup": 2, "orange": 1, "bowl": 1}
        ),
        ShelfZone(
            zone_id=2,
            name="Shelf 2 (Middle)",
            y_min_pct=0.33,
            y_max_pct=0.66,
            expected_inventory={"book": 6}
        ),
        ShelfZone(
            zone_id=3,
            name="Shelf 3 (Bottom)",
            y_min_pct=0.66,
            y_max_pct=1.0,
            expected_inventory={"cup": 2, "bowl": 2}
        ),
    ]

    def __init__(self, model_weights="yolo11n.pt", conf_threshold=0.35, shelves=None):
        self.conf_threshold = conf_threshold
        self.shelves = shelves if shelves else self.DEFAULT_SHELVES
        print(f"[INFO] Loading YOLO model: {model_weights}...")
        self.model = YOLO(model_weights)

    def process_image(self, image_path, output_dir="output", show_display=False):
        """Processes an input shelf image, computes inventory, detects missing stock, and exports reports."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image not found at: {image_path}")

        os.makedirs(output_dir, exist_ok=True)
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to read image file: {image_path}")

        h, w, _ = img.shape
        print(f"[INFO] Processing image '{image_path}' ({w}x{h} px)...")

        # Perform object detection
        results = self.model(img, conf=self.conf_threshold)[0]

        # Structure to track detected items per shelf zone
        detected_by_shelf = {shelf.zone_id: [] for shelf in self.shelves}
        unassigned_detections = []

        annotated_img = img.copy()

        # Process detected bounding boxes
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < self.conf_threshold:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            class_name = self.model.names[cls_id]

            # Calculate object center point
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Find matching shelf zone
            assigned_zone = None
            for shelf in self.shelves:
                if shelf.contains_point(cy, h):
                    assigned_zone = shelf
                    break

            detection_info = {
                "class": class_name,
                "confidence": round(conf, 3),
                "bbox": [x1, y1, x2, y2],
                "center": [cx, cy],
                "shelf_id": assigned_zone.zone_id if assigned_zone else None,
                "shelf_name": assigned_zone.name if assigned_zone else "Unassigned"
            }

            if assigned_zone:
                detected_by_shelf[assigned_zone.zone_id].append(detection_info)
            else:
                unassigned_detections.append(detection_info)

            # Draw item bounding box on annotated image
            color = (0, 215, 255)  # Gold/Yellow for detected items
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
            cv2.circle(annotated_img, (cx, cy), 4, (0, 0, 255), -1)

            label = f"{class_name} {conf:.2f}"
            cv2.putText(
                annotated_img, label, (x1, max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
            )

        # Inventory Audit and Alert Engine
        shelf_reports = []
        overall_restock_required = False

        for shelf in self.shelves:
            detections = detected_by_shelf[shelf.zone_id]
            actual_counts = {}
            for d in detections:
                cls = d["class"]
                actual_counts[cls] = actual_counts.get(cls, 0) + 1

            missing_items = {}
            shelf_restock_needed = False

            for item, expected_qty in shelf.expected_inventory.items():
                actual_qty = actual_counts.get(item, 0)
                if actual_qty < expected_qty:
                    deficit = expected_qty - actual_qty
                    missing_items[item] = {
                        "expected": expected_qty,
                        "present": actual_qty,
                        "missing_count": deficit
                    }
                    shelf_restock_needed = True

            if shelf_restock_needed:
                overall_restock_required = True

            shelf_report = {
                "shelf_id": shelf.zone_id,
                "shelf_name": shelf.name,
                "status": "RESTOCK_REQUIRED" if shelf_restock_needed else "FULLY_STOCKED",
                "expected_inventory": shelf.expected_inventory,
                "present_inventory": actual_counts,
                "missing_items": missing_items,
                "detected_objects": [d["class"] for d in detections]
            }
            shelf_reports.append(shelf_report)

        # Overlay Shelf Zones and Alerts on Image
        overlay = annotated_img.copy()

        for shelf in self.shelves:
            y_min = int(shelf.y_min_pct * h)
            y_max = int(shelf.y_max_pct * h)

            report = next(r for r in shelf_reports if r["shelf_id"] == shelf.zone_id)
            is_alert = report["status"] == "RESTOCK_REQUIRED"

            # Color coding: Green if stocked, Red if restock needed
            zone_color = (0, 0, 220) if is_alert else (0, 180, 0)

            # Draw zone boundary line
            cv2.line(annotated_img, (0, y_max), (w, y_max), zone_color, 2)

            # Draw semi-transparent header bar for shelf zone
            cv2.rectangle(overlay, (0, y_min), (w, y_min + 30), zone_color, -1)
            cv2.addWeighted(overlay, 0.35, annotated_img, 0.65, 0, annotated_img)

            # Zone label & status text
            status_str = f"ALERT: RESTOCK NEEDED" if is_alert else "STATUS: FULLY STOCKED"
            zone_label = f"[{shelf.name}] - {status_str}"
            cv2.putText(
                annotated_img, zone_label, (12, y_min + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2
            )

        # Draw Global Alert Banner on Top of Image
        banner_color = (0, 0, 180) if overall_restock_required else (0, 140, 0)
        cv2.rectangle(annotated_img, (0, 0), (w, 38), banner_color, -1)
        global_status_text = (
            "!!! STOCK RESTORING ALERT: RESTOCKING REQUIRED !!!"
            if overall_restock_required
            else "SHELF INVENTORY STATUS: ALL SHELVES FULLY STOCKED"
        )
        cv2.putText(
            annotated_img, global_status_text, (15, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )

        # Save Annotated Output Image
        output_image_path = os.path.join(output_dir, "annotated_shelf.jpg")
        cv2.imwrite(output_image_path, annotated_img)
        print(f"[SUCCESS] Saved annotated image to: {output_image_path}")

        # Construct JSON & Text Reports
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        alert_summary = {
            "timestamp": timestamp,
            "overall_status": "RESTOCK_REQUIRED" if overall_restock_required else "OK",
            "image_processed": os.path.basename(image_path),
            "shelves": shelf_reports
        }

        # Save JSON Alert Payload
        json_report_path = os.path.join(output_dir, "restock_alerts.json")
        with open(json_report_path, "w", encoding="utf-8") as f:
            json.dump(alert_summary, f, indent=2)
        print(f"[SUCCESS] Saved JSON alert log to: {json_report_path}")

        # Save Human-Readable Text Report
        text_report_path = os.path.join(output_dir, "stock_report.txt")
        with open(text_report_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write(f"SMART SHELF INVENTORY & RESTOCK ALERT REPORT\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Overall Status: {alert_summary['overall_status']}\n")
            f.write("=" * 60 + "\n\n")

            for s in shelf_reports:
                f.write(f"SHELF: {s['shelf_name']} (ID: {s['shelf_id']})\n")
                f.write(f"  Status: {s['status']}\n")
                f.write(f"  Detected Items: {s['present_inventory']}\n")
                f.write(f"  Expected Items: {s['expected_inventory']}\n")
                if s["missing_items"]:
                    f.write("  Restock Alerts:\n")
                    for item, details in s["missing_items"].items():
                        f.write(
                            f"    - MISSING {details['missing_count']}x '{item}' "
                            f"(Found {details['present']} / Expected {details['expected']})\n"
                        )
                else:
                    f.write("  Restock Alerts: None (Fully Stocked)\n")
                f.write("-" * 40 + "\n")

        print(f"[SUCCESS] Saved readable report to: {text_report_path}")

        # Print console output
        self._print_console_summary(alert_summary)

        # GUI Display if enabled
        if show_display:
            cv2.imshow("Smart Shelf Restock Alert Monitor", annotated_img)
            print("[INFO] Press any key on the image window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return alert_summary

    def _print_console_summary(self, summary):
        """Prints formatted alert summary to terminal."""
        print("\n" + "=" * 60)
        print("           SMART SHELF RESTOCK ALERT SYSTEM")
        print("=" * 60)
        print(f" Timestamp:      {summary['timestamp']}")
        print(f" Overall Status: {summary['overall_status']}")
        print("-" * 60)

        for shelf in summary["shelves"]:
            status_flag = "[ALERT]" if shelf["status"] == "RESTOCK_REQUIRED" else "[OK]   "
            print(f"{status_flag} {shelf['shelf_name']}")
            print(f"       Present:  {shelf['present_inventory']}")
            print(f"       Expected: {shelf['expected_inventory']}")
            if shelf["missing_items"]:
                print("       --> MISSING ITEMS FOR RESTOCK:")
                for item, info in shelf["missing_items"].items():
                    print(
                        f"           * {item.upper()}: Need {info['missing_count']} more "
                        f"(Found {info['present']}/{info['expected']})"
                    )
            print("-" * 60)
        print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Smart Shelf Restock Alert System using YOLO & OpenCV")
    parser.add_argument("--image", type=str, default="input/shelf.jpg", help="Path to input shelf image")
    parser.add_argument("--weights", type=str, default="yolo11n.pt", help="YOLO model weights file")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory for output results")
    parser.add_argument("--headless", action="store_true", help="Run without opening GUI window")

    args = parser.parse_args()

    monitor = SmartShelfMonitor(model_weights=args.weights, conf_threshold=args.conf)
    monitor.process_image(
        image_path=args.image,
        output_dir=args.output_dir,
        show_display=not args.headless
    )


if __name__ == "__main__":
    main()
