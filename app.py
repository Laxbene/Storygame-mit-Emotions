import streamlit as st
import cv2
from ultralytics import YOLO
from deepface import DeepFace
import numpy as np

# Seite konfigurieren
st.set_page_config(page_title="AI Emotion Story Game", page_icon="🎭", layout="centered")

# KI-Modelle laden und cachen, damit der Server nicht bei jedem Klick neu lädt
@st.cache_resource
def load_models():
    # Lädt dein hochgeladenes YOLOv12-Face-Modell aus dem Ordner
    face_model = YOLO("models/yolov12n-face(1).pt")
    return face_model

try:
    face_model = load_models()
except Exception as e:
    st.error("Fehler beim Laden des YOLO-Modells. Stelle sicher, dass es im Ordner 'models/' liegt.")

# Spielstand (Story-State) initialisieren
if "story_step" not in st.session_state:
    st.session_state.story_step = 1

st.title("🎭 Das Geheimnis des Tempels")
st.subheader("Ein interaktives Story-Game gesteuert durch deine Emotionen")

# Story-Logik
if st.session_state.story_step == 1:
    st.markdown("""
    ### Kapitel 1: Das verschlossene Tor
    Du stehst vor einem riesigen, alten Tempeltor. 
    *\"Nur wer mit wahrer **Freude** im Herzen herantritt, dem wird sich der Weg öffnen.\"*
    
    **Deine Aufgabe:** Schaue in die Kamera und lächle glücklich (`happy`), um das Tor zu öffnen!
    """)
    target_emotion = "happy"

elif st.session_state.story_step == 2:
    st.markdown("""
    ### Kapitel 2: Der Wächter der Finsternis
    Das Tor öffnet sich krachend! Doch plötzlich springt ein riesiger Schattenwächter vor dich. 
    Du musst ihm zeigen, dass du keine Angst hast, sondern **wütend** und entschlossen bist!
    
    **Deine Aufgabe:** Schaue grimmig und wütend (`angry`) in die Kamera!
    """)
    target_emotion = "angry"

else:
    st.balloons()
    st.success("🎉 Glückwunsch! Du hast das Spiel durch die Kraft deiner Emotionen gemeistert!")
    if st.button("Spiel neustarten"):
        st.session_state.story_step = 1
        st.rerun()
    st.stop()

st.write("---")

# Kamera-Input (Streamlit holt sich hier automatisch die Webcam-Rechte im Browser des Spielers)
img_file_buffer = st.camera_input("Schau in die Kamera für die KI-Erkennung:")

if img_file_buffer is not None:
    # Bild in OpenCV-Format konvertieren
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    # Schritt 1: Gesicht finden mit deinem YOLOv12-Face Modell
    results = face_model(cv2_img, verbose=False)
    boxes = results[0].boxes
    
    if len(boxes) == 0:
        st.warning("🤖 Ich kann dein Gesicht nicht sehen. Bitte positioniere dich besser vor der Kamera.")
    else:
        # Box des ersten erkannten Gesichts ausschneiden
        box = boxes[0].xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = box
        face_crop = cv2_img[y1:y2, x1:x2]
        
        if face_crop.size > 0:
            try:
                # Schritt 2: Emotion auf dem Ausschnitt erkennen
                analysis = DeepFace.analyze(face_crop, actions=['emotion'], enforce_detection=False)
                dominant_emotion = analysis[0]['dominant_emotion']
                emotion_confidence = analysis[0]['emotion'][dominant_emotion]
                
                # Übersetzung für die Spieler
                translations = {
                    "happy": "Fröhlich 😊", "angry": "Wütend 😡", "sad": "Traurig 😢", 
                    "fear": "Ängstlich 😨", "surprise": "Überrascht 😲", "neutral": "Neutral 😐"
                }
                translated_emotion = translations.get(dominant_emotion, dominant_emotion)
                
                st.info(f"Erkannte Emotion: **{translated_emotion}** (Sicherheit: {emotion_confidence:.1f}%)")
                
                # Prüfen, ob die Emotion zur Aufgabe passt
                if dominant_emotion == target_emotion and emotion_confidence > 40:
                    st.success(f"✅ Richtig! Du hast die Emotion '{translated_emotion}' erfolgreich eingesetzt!")
                    if st.button("Weiter zum nächsten Kapitel"):
                        st.session_state.story_step += 1
                        st.rerun()
                else:
                    st.error(f"❌ Versuche es noch einmal. Gesucht ist: **{target_emotion}**.")
                    
            except Exception as e:
                st.error("Fehler bei der Emotionsanalyse. Versuche, dich deutlicher zu zeigen.")
