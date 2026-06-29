import streamlit as st
import cv2
from ultralytics import YOLO
from deepface import DeepFace
import numpy as np
from PIL import Image

# 1. Seite konfigurieren
st.set_page_config(page_title="AI Emotion Story Game", page_icon="🎭", layout="centered")

# 2. KI-Modelle laden (wird gecached, damit es schnell geht)
@st.cache_resource
def load_models():
    # Lädt dein hochgeladenes YOLOv12-Face-Modell
    face_model = YOLO("models/yolov12n-face(1).pt")
    return face_model

try:
    face_model = load_models()
except Exception as e:
    st.error(f"Fehler beim Laden des YOLO-Modells. Überprüfe den Pfad 'models/yolov12n-face(1).pt'.")

# 3. Spielstand (Story-State) initialisieren
if "story_step" not in st.session_state:
    st.session_state.story_step = 1
if "game_success" not in st.session_state:
    st.session_state.game_success = False

st.title("🎭 Das Geheimnis des Tempels")
st.subheader("Ein interaktives Story-Game gesteuert durch deine Emotionen")

# 4. Story-Logik definieren
if st.session_state.story_step == 1:
    st.markdown("""
    ### Kapitel 1: Das verschlossene Tor
    Du stehst vor einem riesigen, alten Tempeltor. Eingraviert ist eine alte Runenschrift: 
    *\"Nur wer mit wahrer **Freude** im Herzen herantritt, dem wird sich der Weg öffnen.\"*
    
    **Deine Aufgabe:** Schaue in die Kamera und lächle glücklich (`happy`), um das Tor zu öffnen!
    """)
    target_emotion = "happy"

elif st.session_state.story_step == 2:
    st.markdown("""
    ### Kapitel 2: Der Wächter der Finsternis
    Das Tor öffnet sich krachend! Doch plötzlich springt ein riesiger Schattenwächter vor dich. 
    Er will dich einschüchtern. Du musst ihm zeigen, dass du keine Angst hast, sondern **wütend** und entschlossen bist!
    
    **Deine Aufgabe:** Schaue grimmig und wütend (`angry`) in die Kamera!
    """)
    target_emotion = "angry"

else:
    st.balloons()
    st.success("🎉 Glückwunsch! Du hast das Spiel durch die Kraft deiner Emotionen gemeistert!")
    if st.button("Spiel neustarten"):
        st.session_state.story_step = 1
        st.session_state.game_success = False
        st.rerun()
    st.stop()


# 5. Kamera-Input von Streamlit nutzen
st.write("---")
img_file_buffer = st.camera_input("Schau in die Kamera für die KI-Erkennung:")

if img_file_buffer is not None:
    # Bild in ein OpenCV-Format konvertieren
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    # Schritt A: Gesicht finden mit deinem YOLOv12-Face Modell
    results = face_model(cv2_img, verbose=False)
    
    boxes = results[0].boxes
    if len(boxes) == 0:
        st.warning("🤖 Ich kann dein Gesicht nicht sehen. Bitte positioniere dich besser vor der Kamera.")
    else:
        # Wenn ein Gesicht gefunden wurde, nehmen wir die Box des ersten Gesichts
        box = boxes[0].xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = box
        
        # Gesicht ausschneiden (Cropping) für die Emotionserkennung
        face_crop = cv2_img[y1:y2, x1:x2]
        
        if face_crop.size > 0:
            try:
                # Schritt B: Emotion auf dem ausgeschnittenen Gesicht erkennen mit DeepFace
                # (enforce_detection=False, da YOLO das Gesicht bereits gefunden hat)
                analysis = DeepFace.analyze(face_crop, actions=['emotion'], enforce_detection=False)
                
                # DeepFace gibt eine Liste zurück
                dominant_emotion = analysis[0]['dominant_emotion']
                emotion_confidence = analysis[0]['emotion'][dominant_emotion]
                
                # Übersetzung für die UI
                translations = {"happy": "Fröhlich 😊", "angry": "Wütend 😡", "sad": "Traurig 😢", 
                                "fear": "Ängstlich 😨", "surprise": "Überrascht 😲", "neutral": "Neutral 😐", "disgust": "Ekel 🤢"}
                
                translated_emotion = translations.get(dominant_emotion, dominant_emotion)
                
                st.info(f"Erkannte Emotion: **{translated_emotion}** (Sicherheit: {emotion_confidence:.1f}%)")
                
                # Prüfen, ob die Emotion zur Story passt
                if dominant_emotion == target_emotion and emotion_confidence > 40:
                    st.success(f"✅ Richtig! Du hast die Emotion '{translated_emotion}' erfolgreich eingesetzt!")
                    if st.button("Weiter zum nächsten Kapitel"):
                        st.session_state.story_step += 1
                        st.rerun()
                else:
                    st.error(f"❌ Das reicht noch nicht. Gesucht ist: **{target_emotion}**. Du zeigst aktuell: {dominant_emotion}.")
                    
            except Exception as e:
                st.error("Fehler bei der Emotionsanalyse. Versuche, dich deutlicher zu zeigen.")
