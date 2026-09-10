# Smart Shelf Restock Alert System 🛒📦

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![YOLOv11](https://img.shields.io/badge/YOLO-v11-orange.svg)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)

An intelligent, AI-powered computer vision system designed for **retail store shelves, warehouse racks, and inventory monitoring**. Built with **YOLOv11** and **OpenCV**, this project tracks items across specific shelf zones, audits current inventory against expected stock levels, and automatically triggers **Stock Restoring Alerts** when items are missing or depleted.

---

## 🌟 Key Features

- 🎯 **Zone-Based Multi-Shelf Monitoring**: Divides shelf images into customizable Regions of Interest (ROIs) corresponding to specific physical shelves (Top, Middle, Bottom).
- 🤖 **YOLOv11 Object Detection**: Detects products (bottles, cups, books, cans, etc.) with high precision and confidence scoring.
- 🚨 **Automated Restock Alert Engine**: Compares detected items per shelf against target expected counts. Instantly flags missing inventory.
- 🎨 **Rich OpenCV Visual Overlays**:
  - Color-coded shelf zone headers (Green = Fully Stocked, Red = Restock Needed).
  - Bounding box object highlights and center-point mapping.
  - On-Screen Display (OSD) alert header banner.
- 📊 **Multi-Format Exporting**:
  - `output/annotated_shelf.jpg`: Annotated visual output with alert banners.
  - `output/restock_alerts.json`: Machine-readable structured alert logs for API integrations.
  - `output/stock_report.txt`: Human-readable summary report for store staff.
- ⚡ **Headless & Interactive Modes**: Seamlessly runs on headless servers, edge devices (Raspberry Pi/NVIDIA Jetson), or interactive desktop apps.

---

## 📂 Project Structure

```text
smart_shelf_restock_alert/
├── smart_shelf_monitor.py   # Main inventory audit & alert pipeline
├── generate_demo.py         # Synthetic shelf image generator for testing
├── requirements.txt         # Dependency declarations
├── README.md                # Project documentation
├── input/
│   └── shelf.jpg            # Input shelf image
└── output/
    ├── annotated_shelf.jpg  # Visual output with alert overlays
    ├── restock_alerts.json  # Exported JSON alert data
    └── stock_report.txt     # Text summary report
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure Python 3.9+ is installed on your system.

### 2. Installation
Clone the repository and install the dependencies:

```bash
git clone https://github.com/your-username/smart-shelf-restock-alert.git
cd smart-shelf-restock-alert

pip install -r requirements.txt
```

---

## ⚡ Quickstart

### Step 1: Generate a Sample Test Image
Run the generator script to create a sample multi-shelf image (`input/shelf.jpg`) with items placed across top, middle, and bottom shelves:

```bash
python generate_demo.py
```

### Step 2: Run the Smart Shelf Monitor

**Run in Headless Mode (saves output files to `output/`):**
```bash
python smart_shelf_monitor.py --image input/shelf.jpg --headless
```

**Run in Interactive GUI Mode (opens OpenCV image window):**
```bash
python smart_shelf_monitor.py --image input/shelf.jpg
```

---

## ⚙️ Configuration & Customization

You can customize the shelf zones and expected inventory directly in `smart_shelf_monitor.py` by defining `ShelfZone` objects:

```python
ShelfZone(
    zone_id=1,
    name="Shelf 1 (Top)",
    y_min_pct=0.0,    # Top boundary (0% of image height)
    y_max_pct=0.33,   # Bottom boundary (33% of image height)
    expected_inventory={"bottle": 2, "cup": 1} # Target items
)
```

---

## 📋 Sample Output

### 1. Terminal Console Summary
```text
============================================================
           SMART SHELF RESTOCK ALERT SYSTEM
============================================================
 Timestamp:      2026-09-10 20:25:30
 Overall Status: RESTOCK_REQUIRED
------------------------------------------------------------
[ALERT] Shelf 1 (Top)
       Present:  {'bottle': 1}
       Expected: {'bottle': 2, 'cup': 1}
       --> MISSING ITEMS FOR RESTOCK:
           * BOTTLE: Need 1 more (Found 1/2)
           * CUP: Need 1 more (Found 0/1)
------------------------------------------------------------
[OK]    Shelf 2 (Middle)
       Present:  {'book': 2, 'bottle': 1}
       Expected: {'book': 2, 'bottle': 1}
------------------------------------------------------------
[ALERT] Shelf 3 (Bottom)
       Present:  {'bottle': 1, 'cup': 1}
       Expected: {'bottle': 2, 'cup': 1, 'book': 1}
       --> MISSING ITEMS FOR RESTOCK:
           * BOTTLE: Need 1 more (Found 1/2)
           * BOOK: Need 1 more (Found 0/1)
============================================================
```

### 2. JSON Output (`output/restock_alerts.json`)
```json
{
  "timestamp": "2026-09-10 20:25:30",
  "overall_status": "RESTOCK_REQUIRED",
  "image_processed": "shelf.jpg",
  "shelves": [
    {
      "shelf_id": 1,
      "shelf_name": "Shelf 1 (Top)",
      "status": "RESTOCK_REQUIRED",
      "missing_items": {
        "bottle": { "expected": 2, "present": 1, "missing_count": 1 },
        "cup": { "expected": 1, "present": 0, "missing_count": 1 }
      }
    }
  ]
}
```

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
