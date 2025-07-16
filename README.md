# 👁️ Face Attendance System with Liveness Detection & Telegram Alerts

A smart facial recognition attendance and access system that includes **liveness detection** via eye movement, real-time face verification, and automatic Telegram alerts.

---

## 🚀 Features

- 🔒 Real-time face recognition using `face_recognition`
- 👁️ Blink-based liveness detection using `dlib`'s facial landmarks
- 📸 Live webcam video feed with visual feedback
- 📬 Sends Telegram alerts when access is granted
- 📝 Logs successful entries in a CSV file with name, date, and time
- 🧠 Handles multiple faces, cooldowns, and unknown users

---

## 🛠️ Tech Stack

- **Python 3.9**
- **OpenCV** – Real-time webcam feed and overlays
- **dlib** – Facial landmark detection
- **face_recognition** – Face encoding and recognition
- **NumPy** – Landmark and EAR calculations
- **Requests** – Telegram API messaging
- **CSV** – Attendance logging

---

## 📁 Folder Structure
<pre>
  face_attendance/
├── authorized_faces/
├── access_log.csv (auto-generated)
├── shape_predictor_68_face_landmarks.dat
├── link_to_install.txt
├── live_recognition.py
├── LICENSE
└── README.md
</pre>

---
## 🛠️ Installation

```bash
git clone https://github.com/daksheshsharma2409/face_attendance.git
cd face_attendance
pip install -r requirements.txt
python live_recognition.py
```
---

## 🧪 Setup Instructions

### 1. Add Authorized Faces

- Place clear, front-facing images of people inside the `authorized_faces/` folder.
- The image name (without extension) is used as the label.
  - Example: `john_doe.jpg` → `John Doe`

### 2. Dlib Shape Predictor

Download this file:

🔗 http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

- Unzip it.
- Place the `.dat` file in the root directory.
- A text file `link_to_install.txt` is included with the download link.

### 3. Configure Telegram Alerts

In `live_recognition.py`, replace the placeholders:

```python
TELEGRAM_BOT_TOKEN = 'Enter BOT Token'
TELEGRAM_CHAT_ID = 'Enter Chat_ID'
```
Follow Telegram’s bot guide to generate a token and get your chat ID.

🔗 https://core.telegram.org/bots

---

## ▶️ How to Run

```bash
python live_recognition.py
```
- Make sure your webcam is connected.

- The system will wait until a single face is detected.

- Once detected, blink when prompted.

- If successful, face recognition is triggered.

- On success, access is granted, logged, and a Telegram alert is sent.

- Press q to exit anytime.

---

## 📊 Output

- ✅ **Green box** – Access granted
- ❌ **Red box** – Access denied
- ⚠️ **Yellow box** – Waiting for face or multiple faces detected
- 👁️ **Blink detection** and **EAR (Eye Aspect Ratio)** shown during liveness
- 📤 **Telegram alert**: Name, time, and date of successful access sent to your Telegram bot

---

## 🧠 Notes

- Works **fully offline** except for Telegram integration
- Ensure only **one face** is visible during verification to avoid false triggers
- Dlib shape predictor file should be placed as instructed in `link_to_install.txt`

---

## 👨‍💻 Author

**Made with ❤️ by Dakshesh Sharma**  
[GitHub Profile](https://github.com/daksheshsharma2409)

---

## 📜 License

This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for more details.

---

