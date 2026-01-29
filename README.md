# Face Recognition Attendance System 🟢

A **Python desktop attendance system** using **face recognition**. Detects and recognizes **only registered users** and logs attendance with **single punch-in and punch-out**. Built with **Tkinter**, **OpenCV**, **MTCNN**, and **FaceNet**.

---

## Features

- ✅ Recognizes **only registered users**  
- ❌ Rejects **unknown/unregistered faces**  
- 🕑 Single **Punch-In / Punch-Out** per session  
- 💻 **Tkinter GUI** with live webcam feed  
- 📊 Attendance logged in **CSV** format (`data/attendance.csv`)  

---

## Folder Structure

---

## Requirements

Install Python 3.10+ and the required packages:

```bash
pip install opencv-python numpy mtcnn keras-facenet Pillow tensorflow
python app.py
