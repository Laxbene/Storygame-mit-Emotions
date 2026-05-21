import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

# --- KONFIGURATION ---
# Falls dein Modell in einem Unterordner liegt, passe den Pfad hier an.
# Wenn die Datei direkt im selben Ordner wie app.py liegt, reicht "yolov12n-face.pt"
MODEL_PATH = "ki_modell/yolov12n-face.pt" 

# Falls der Ordner anders heißt, kannst du ihn hier dynamisch suchen:
if not os.path.exists(MODEL_PATH):
    # Fallback, falls das Modell im Hauptverzeichnis liegt
    if os.path.exists("yolov12n-face.pt"):
        MODEL_PATH = "yolov12n-face.pt"

# --- KI MODELL LADEN ---
@st.cache_resource
def load_yolo_model(path):
    try:
        from ultralytics import YOLO
        if os.path.exists(path):
            return YOLO(path)
        else:
            return None
    except ImportError:
        return None

model = load_yolo_model(MODEL_PATH)

# --- STREAMLIT UI ---
st.set_page_config(
    page_title="YOLOv12 Gesichtserkennung",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 YOLOv12 Gesichtserkennung")
st.write("Lade ein Bild hoch, und die KI wird automatisch Gesichter darin erkennen.")

# Überprüfung, ob das Modell und die Library korrekt geladen wurden
if model is None:
    st.error(f"Fehler: Das Modell konnte unter '{MODEL_PATH}' nicht gefunden werden oder `ultralytics` ist nicht installiert.")
    st.info("Bitte stelle sicher, dass die Datei `yolov12n-face.pt` im richtigen Ordner liegt und `ultralytics` in den Requirements steht.")
else:
    st.success("YOLOv12-Modell erfolgreich geladen!")

    # Datei-Uploader für den Nutzer
    uploaded_file = st.file_uploader("Wähle ein Bild aus...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Bild mit PIL öffnen
        image = Image.open(uploaded_file)
        
        # Streamlit Spalten für Vorher/Nachher
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Originalbild")
            st.image(image, use_container_width=True)
            
        with col2:
            st.subheader("KI-Erkennung")
            
            # Vorhersage starten, wenn der Button geklickt wird
            if st.button("Gesichter erkennen", type="primary"):
                with st.spinner("Modell analysiert das Bild..."):
                    # Konvertiere PIL-Bild zu einem Format, das YOLO versteht (OpenCV/NumPy)
                    img_array = np.array(image)
                    
                    # KI-Vorhersage ausführen
                    results = model(img_array)
                    
                    # Ergebnisse auf dem Bild einzeichnen (.plot() gibt ein BGR-Bild zurück)
                    res_plotted = results[0].plot()
                    
                    # Konvertiere BGR zurück zu RGB für Streamlit
                    res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                    
                    # Anzahl der erkannten Gesichter auslesen
                    num_faces = len(results[0].boxes)
                    
                    # Ergebnis anzeigen
                    st.image(res_rgb, use_container_width=True)
                    
                    if num_faces > 0:
                        st.metric(label="Erkannte Gesichter", value=num_faces)
                    else:
                        st.warning("Keine Gesichter im Bild gefunden.")import streamlit as st
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
